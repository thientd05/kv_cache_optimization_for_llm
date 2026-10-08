"""How much to trust a single bench.py run, and why.

bench.py checks whether the mean normalised latency has settled by comparing the first half of
the run against the second (it stands in for the paper's 1-hour traces, which we cannot match
on this GPU). But a large gap between the halves has two completely different causes, and
treating them the same is a trap:

  * the run was too short, so the mean is still noisy          -> re-run it longer
  * the system is past its capacity point, so the queue grows
    without bound and latency genuinely diverges for as long
    as the run lasts                                           -> the data is CORRECT, and a
                                                                  longer run makes the gap worse

The second case is not a defect, it is the result: it is what makes the latency curve explode
in the paper's Fig. 12, and those points are exactly what locates the knee. Deleting and
re-running them would throw away the finding and waste hours.

The discriminator is the queue. Below capacity the queue stays empty and any drift is noise.
Past capacity the queue builds up monotonically, which bench.py records as waiting_mean.
"""

# Queue depth above which the arrival rate is clearly outrunning the server. The engine samples
# telemetry every STEP_TELEMETRY_EVERY decode steps, so this is a time-average over the run: a
# mean of 0.3 already means the queue is rarely empty.
SATURATION_WAITING = 0.3
DRIFT_TOLERANCE = 0.10
MIN_SAMPLES = 20

STEADY = "steady"
SATURATED = "saturated"
NOISY = "noisy"
BROKEN = "broken"

LABEL_VI = {
    STEADY: "ổn định",
    SATURATED: "quá tải",
    NOISY: "run quá ngắn",
    BROKEN: "LỖI",
}


def classify(run):
    """-> (kind, one-line explanation in Vietnamese).

    `run` is a parsed bench.py result dict.
    """
    if run.get("abandoned"):
        return BROKEN, f"run bị bỏ dở: {run['abandoned']}"
    if not run.get("invariants", {}).get("ok", True):
        inv = run.get("invariants", {})
        return BROKEN, (f"vi phạm bất biến token — budget_mismatch={inv.get('budget_mismatch')}, "
                        f"engine_mismatch={inv.get('engine_mismatch')}. Đây là lỗi thật.")

    conv = run.get("convergence", {})
    drift = conv.get("drift", 1.0)
    waiting = run.get("waiting_mean", 0.0)
    completed = run.get("completed", 0)

    if completed < MIN_SAMPLES:
        return NOISY, f"chỉ {completed} request hoàn tất, quá ít để lấy trung bình"
    if drift < DRIFT_TOLERANCE:
        return STEADY, f"nửa đầu và nửa sau lệch {drift:.0%}, trung bình đã ổn định"

    first = conv.get("first_half", 0.0)
    second = conv.get("second_half", 0.0)
    rising = second > first

    # Saturation needs the QUEUE, not just a rising mean. An earlier version accepted either
    # signal, which mislabelled runs that drifted upward on burst sensitivity alone - the queue
    # was empty on average, so the server was keeping up and the mean simply had not settled.
    if waiting >= SATURATION_WAITING and rising:
        return SATURATED, (f"hàng đợi trung bình {waiting:.1f} request, latency tăng "
                           f"{first:.3f} → {second:.3f} s/token trong run. Hệ thống vượt năng lực "
                           f"— đây là kết quả đúng, và con số báo cáo là CẬN DƯỚI của latency thật "
                           f"ở trạng thái ổn định")
    if rising:
        return NOISY, (f"lệch {drift:.0%} và đang TĂNG ({first:.3f} → {second:.3f} s/token) nhưng "
                       f"hàng đợi vẫn gần trống ({waiting:.2f}) — trung bình chưa ổn định, có thể do "
                       f"nhạy với cụm request đến dồn. Con số là cận dưới; nếu đây là biến thể CHẬM "
                       f"hơn thì sai lệch theo hướng bảo thủ, còn muốn số chắc thì chạy lại với "
                       f"--num-requests lớn hơn")
    return NOISY, (f"lệch {drift:.0%} theo hướng GIẢM ({first:.3f} → {second:.3f} s/token), hàng đợi "
                   f"trống ({waiting:.2f}) — nhiễu hai chiều, chạy lại với --num-requests lớn hơn")


def is_trustworthy(run):
    """Steady and saturated runs both belong in the curve; only noisy and broken ones do not.
    A saturated point understates its own latency, which is conservative, not misleading."""
    return classify(run)[0] in (STEADY, SATURATED)


def needs_attention(run):
    return classify(run)[0] in (NOISY, BROKEN)

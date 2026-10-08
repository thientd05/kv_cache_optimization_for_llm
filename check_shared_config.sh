#!/usr/bin/env bash
# The two builds keep their shared constants as a duplicated block inside their own src/config.h,
# between the BEGIN/END SHARED CONFIG markers. Nothing in the compiler enforces that the two copies
# agree, so this does: it diffs them and fails loudly if they have drifted.
#
# Run it after touching either config.h. Both build.sh scripts call it first.
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

extract() { sed -n '/BEGIN SHARED CONFIG/,/END SHARED CONFIG/p' "$1"; }

a="$here/base/src/config.h"
b="$here/paged_attention/src/config.h"

for f in "$a" "$b"; do
    if ! grep -q 'BEGIN SHARED CONFIG' "$f" || ! grep -q 'END SHARED CONFIG' "$f"; then
        echo "FAIL: $f is missing its BEGIN/END SHARED CONFIG markers" >&2
        exit 1
    fi
done

if diff -u <(extract "$a") <(extract "$b") > /tmp/shared_config.diff; then
    echo "shared config OK: base and paged_attention agree ($(extract "$a" | wc -l) lines)"
    rm -f /tmp/shared_config.diff
    exit 0
fi

echo "FAIL: the shared config block has drifted between the two builds." >&2
echo "Left = base/src/config.h, right = paged_attention/src/config.h:" >&2
cat /tmp/shared_config.diff >&2
echo >&2
echo "Fix: make the two blocks identical again. A value that is genuinely allowed to differ" >&2
echo "belongs BELOW the END marker, not above it." >&2
exit 1

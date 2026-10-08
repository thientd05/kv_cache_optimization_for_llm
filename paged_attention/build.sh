set -e
# base/src/config.h and paged_attention/src/config.h carry a duplicated shared-config block that
# nothing in the compiler keeps in sync. Refuse to build a comparison whose two sides disagree.
"$(dirname "$0")/../check_shared_config.sh"

rm -rf build/
mkdir -p build
cd build && cmake .. -G Ninja && ninja

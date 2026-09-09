#!/usr/bin/env bash
# Run in a terminal as the normal user; sudo is used only for diagnostic reads.
set -euo pipefail

panic_output="${1:-checkpoints/training_debug/kernel-panic-$(date +%Y%m%d-%H%M%S)}"
sudo -v
mkdir -p "$panic_output"
sudo journalctl -k -b -1 --no-pager -o short-iso > "$panic_output/previous-kernel.log"
sudo journalctl -k -b 0 --no-pager -o short-iso > "$panic_output/current-kernel.log"
sudo journalctl --list-boots --no-pager > "$panic_output/boots.txt"
sudo tar -C /var/lib/systemd -cf - pstore > "$panic_output/pstore.tar"
sudo dmidecode --type 17 | grep -E 'Memory Device|Size:|Locator:|Type:|Speed:|Configured Memory Speed:|Configured Voltage:' > "$panic_output/memory-settings.txt"
printf 'Saved kernel journals, archived panic logs, and memory settings to %s\n' "$panic_output"

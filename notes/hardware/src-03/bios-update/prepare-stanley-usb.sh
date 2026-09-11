#!/bin/bash
# Erases only the USB explicitly authorized by the user on 2026-09-07.
set -euo pipefail
export PATH=/usr/sbin:/usr/bin:/sbin:/bin

USB_ID=/dev/disk/by-id/usb-SanDisk_USB_Flash_Drive_4C531001391120109300-0:0
EXPECTED_SERIAL=4C531001391120109300
BIOS_DIR=/home/giraf/Documents/GIRAF-Learning/checkpoints/training_debug/src03_bios_20260907
CAP_NAME=ROG-STRIX-Z790-A-GAMING-WIFI-II-ASUS-2202.CAP
CAP_SHA=1f04a9a30b81df6b2dbcf5887b6147a819a387e79e8941394f92de898c8d8d28

die() { printf 'STOP: %s\n' "$*" >&2; exit 1; }
[[ $EUID == 0 ]] || die 'Run this script with sudo from your local terminal.'
[[ $(hostname) == src-03 ]] || die 'This script is only for SRC-03.'
[[ -L "$USB_ID" && -b "$USB_ID" ]] || die 'The authorized SanDisk USB is missing.'
USB_DEVICE=$(readlink -f "$USB_ID")
[[ "$USB_DEVICE" =~ ^/dev/sd[a-z]+$ ]] || die 'Unexpected USB device path.'
[[ $(lsblk -dn -o SERIAL "$USB_DEVICE" | xargs) == "$EXPECTED_SERIAL" ]] || die 'USB serial mismatch.'
[[ $(lsblk -dn -o TRAN "$USB_DEVICE" | xargs) == usb ]] || die 'Target is not a USB drive.'
[[ $(lsblk -dn -o RM "$USB_DEVICE" | xargs) == 1 ]] || die 'Target is not removable.'
USB_BYTES=$(blockdev --getsize64 "$USB_DEVICE")
(( USB_BYTES > 100000000000 && USB_BYTES < 140000000000 )) || die 'Unexpected USB capacity.'
[[ $(cat /sys/class/dmi/id/board_name) == 'ROG STRIX Z790-A GAMING WIFI II' ]] || die 'Motherboard mismatch.'
[[ $(sha256sum "$BIOS_DIR/$CAP_NAME" | cut -d ' ' -f 1) == "$CAP_SHA" ]] || die 'Source firmware hash mismatch.'

unmount_usb_partitions() {
    local part target
    while IFS= read -r part; do
        [[ "$part" == "$USB_DEVICE" ]] && continue
        [[ $(lsblk -dn -o TYPE "$part" | xargs) == part ]] || die 'Unexpected device under USB.'
        while IFS= read -r target; do
            [[ -z "$target" ]] && continue
            case "$target" in
                /media/*|/run/media/*) ;;
                *) die "Unexpected USB mount point: $target" ;;
            esac
            umount -- "$target"
        done < <(findmnt -rn -S "$part" -o TARGET || true)
        [[ -z $(lsblk -dn -o MOUNTPOINTS "$part" | xargs) ]] || die 'A USB partition remains in use.'
    done < <(lsblk -nrpo NAME "$USB_DEVICE")
}

printf 'Authorized erase: %s, serial %s, %s bytes\n' "$USB_DEVICE" "$EXPECTED_SERIAL" "$USB_BYTES"
unmount_usb_partitions
[[ $(readlink -f "$USB_ID") == "$USB_DEVICE" ]] || die 'Device changed before partitioning.'
printf 'label: dos\nunit: sectors\n\nstart=2048, size=16777216, type=c\n' |
    sfdisk --wipe always --wipe-partitions always "$USB_DEVICE"
udevadm settle
USB_PART="${USB_DEVICE}1"
[[ -b "$USB_PART" ]] || die 'New USB partition did not appear.'
unmount_usb_partitions
mkfs.fat -F 32 -n ASUS_BIOS "$USB_PART"
udevadm settle
unmount_usb_partitions

USB_MOUNT=$(mktemp -d /tmp/src03-bios-usb.XXXXXX)
USB_MOUNTED=0
cleanup() {
    if (( USB_MOUNTED )); then umount -- "$USB_MOUNT" || return; fi
    rmdir -- "$USB_MOUNT"
}
trap cleanup EXIT
mount -t vfat -o rw,nosuid,nodev,noexec "$USB_PART" "$USB_MOUNT"
USB_MOUNTED=1
cp -- "$BIOS_DIR/$CAP_NAME" "$USB_MOUNT/$CAP_NAME"
cp -- "$BIOS_DIR/$CAP_NAME" "$USB_MOUNT/A5459.CAP"
cp -- "$BIOS_DIR/README.md" "$USB_MOUNT/README.txt"
sync -f "$USB_MOUNT"
umount -- "$USB_MOUNT"
USB_MOUNTED=0
mount -t vfat -o ro,nosuid,nodev,noexec "$USB_PART" "$USB_MOUNT"
USB_MOUNTED=1
for name in "$CAP_NAME" A5459.CAP; do
    [[ $(sha256sum "$USB_MOUNT/$name" | cut -d ' ' -f 1) == "$CAP_SHA" ]] || die "USB verification failed: $name"
    printf 'Verified USB copy: %s\n' "$name"
done
umount -- "$USB_MOUNT"
USB_MOUNTED=0
printf '\nREADY: ASUS_BIOS has one 8 GiB FAT32 partition on MBR. Remaining space is unallocated.\n'
printf 'The former STANLEY partitions were erased. Both firmware copies passed SHA-256 verification.\n'
printf 'The USB is unmounted. No motherboard firmware or CPU settings were changed.\n'
lsblk -o NAME,SIZE,MODEL,PTTYPE,FSTYPE,LABEL,MOUNTPOINTS "$USB_DEVICE"

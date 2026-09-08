# SRC-03 BIOS update preparation — 2026-09-07

Observed before update:

- Host: src-03
- Board: ASUS ROG STRIX Z790-A GAMING WIFI II (the II model matters)
- CPU: Intel Core i9-14900KS
- BIOS: 1801, firmware build date 2024-11-30
- Linux runtime microcode: 0x133
- Kernel: 7.0.0-31-generic
- CPU turbo enabled (`intel_pstate/no_turbo = 0`)
- Ubuntu boot entry: `\EFI\ubuntu\shimx64.efi` on NVMe EFI partition
- Internal drive: WD_BLACK SN850X HS 2000GB; EFI partition plus ext4 Linux root
- Privileged kernel logs and DIMM details require the user's sudo password.

Target: ASUS BIOS 2202, released 2026-08-27, for this exact board.

Official release page:
https://www.asus.com/uk/supportonly/rog%20strix%20z790-a%20gaming%20wifi%20ii/helpdesk_bios/

Official archive:
https://dlcdnets.asus.com/pub/ASUS/mb/BIOS/ROG-STRIX-Z790-A-GAMING-WIFI-II-ASUS-2202.zip

ASUS-published archive SHA-256:
`e2547e1e6a8d250543904855bd99fc12fe6ec15e5721f0abbb72a24341d5762e`

Preparation completed: downloaded from ASUS, archive hash matches the published
value, ZIP integrity test passed, and CAP extracted. Extracted CAP SHA-256
(locally calculated for verifying a subsequent USB copy):
`1f04a9a30b81df6b2dbcf5887b6147a819a387e79e8941394f92de898c8d8d28`

Firmware has not been flashed. No BIOS or CPU settings have been changed.

ASUS lists an Intel IPU 2026.1 microcode/memory-compatibility update and ME
16.1.40.2765. The BIOS update also updates ME; ASUS says ME stays updated even
if the BIOS is later downgraded. These notes do not certify the CPU is healthy.

Before rebooting:

1. Preserve important data on separate storage and save open work.
2. Identify the USB drive before copying anything. Do not format any drive
   without confirming its identity and whether its existing contents can be erased.
3. Use a compatible FAT32 USB drive, with the extracted CAP file in its root.
4. Verify the archive against the published hash and verify the USB copy
   against the extracted CAP. A ZIP archive itself is not the flash input.
5. Keep this guide accessible on another device during reboot.

In firmware:

1. Restart and press Delete/F2 to enter setup. Photograph current CPU/memory,
   storage/VMD, boot, Secure Boot, and fan/pump settings before changing them.
2. If firmware itself freezes or resets, do not start an EZ Flash update;
   stop and assess the board's USB BIOS FlashBack procedure instead.
3. F7 (Advanced Mode) -> Tool -> ASUS EZ Flash 3 Utility. Select the USB's
   extracted CAP. Confirm the exact board model and target version 2202.
4. Start the update only with stable power. Do not interrupt power, reset,
   remove the USB, or force shutdown while flashing or during automatic restarts.
5. After completion, enter setup and verify BIOS 2202. Load Optimized Defaults
   (F5), then select Intel Default Settings, Performance if offered. Leave XMP
   disabled (Ai Overclock Tuner: Auto), and avoid CPU overclock/undervolt overrides.
6. Preserve any required storage/boot configuration from the photographs;
   keep Ubuntu on the NVMe drive as the boot target and check fan/pump operation.
   Save and boot Ubuntu. Do not reload an old overclocking profile.

EZ Flash uses the extracted CAP filename. ASUS specifies `A5459.CAP` for the
separate rear-button USB BIOS FlashBack procedure; that procedure requires the
designated USB port and different power/button steps. Do not mix the two.

After returning to Ubuntu:

- Recheck BIOS, runtime microcode, memory configuration, and kernel messages.
- Test memory at default settings before another extended training run.
- Re-run the known failing workload with logs, then increase duration if stable.
- Persistent faults at current firmware/defaults need further CPU/RAM isolation;
  a firmware update does not establish whether a processor has already degraded.

Official guidance:

- https://www.asus.com/support/faq/1012815/ (EZ Flash)
- https://www.asus.com/support/faq/1038568/ (USB BIOS FlashBack)
- https://www.intel.com/content/www/us/en/support/articles/000102331/processors.html
  (Intel Default Settings, BIOS microcode 0x12F or later, and extended warranty)

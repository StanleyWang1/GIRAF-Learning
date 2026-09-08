# SRC-03 BIOS update guide — 2026-09-07

## Ready to reboot

USB preparation is complete. Leave the USB plugged in as it is. Restart normally
and press Delete repeatedly as the system starts (F2 is an alternative).
Enter motherboard BIOS setup; do not select the USB as a boot device.
This drive contains firmware files, not a bootable operating system.

Before restarting, save your work, confirm important data is backed up on separate
storage, and keep this guide open on your phone or another computer. The local
README and USB README cannot be read normally from inside BIOS setup.
Use stable power, ideally a UPS. We have not verified an off-PC data backup or UPS.

## USB preparation — completed and verified

- USB: SanDisk, serial 4C531001391120109300, formerly STANLEY.
- User-authorized reformat completed: one 8 GiB FAT32 partition, MBR partition table.
- Label: ASUS_BIOS. The remaining USB capacity is unallocated.
- Old USB partitions/files were removed; no backup of them was made.
- The root directory contains these two byte-identical firmware files:
  - ROG-STRIX-Z790-A-GAMING-WIFI-II-ASUS-2202.CAP — select this in EZ Flash.
  - A5459.CAP — reserved here for the separate USB BIOS FlashBack recovery method.
- Both copies passed SHA-256 verification after unmounting and remounting.
- The USB is unmounted in Linux; this is expected and it can stay plugged in.
- Do not rerun prepare-stanley-usb.sh: that script erases the USB again.

## Update through BIOS setup

1. Enter setup with Delete/F2. Press F7 for Advanced Mode if needed.
2. Photograph Main, Ai Tweaker, boot order, storage/VMD, Secure Boot, and any
   custom fan/pump settings before changing them. Check CPU temperature is not
   rapidly rising toward its thermal limit and that cooling appears functional.
   If setup freezes, resets, or cooling looks faulty, stop BEFORE starting the flash.
3. Open Tool -> ASUS EZ Flash 3 Utility.
4. Select the USB volume and this exact file:
   ROG-STRIX-Z790-A-GAMING-WIFI-II-ASUS-2202.CAP
   The volume may appear as a filesystem identifier rather than ASUS_BIOS.
5. Check the displayed motherboard matches ROG STRIX Z790-A GAMING WIFI II and
   target BIOS version is 2202. If the tool rejects the file or details differ,
   stop; do not bypass the check. Confirm the update when the details match.
6. Once flashing starts, do not reset, shut down, disconnect power, remove the
   USB, or press the rear FlashBack/Clear CMOS buttons. Allow automatic restarts
   and memory initialization to complete; these can take several minutes.
7. When the update has completed, enter BIOS again (Delete/F2, or F1 if prompted)
   and verify version 2202. Press F5 to load Optimized Defaults and confirm.
8. In Ai Tweaker, choose Performance Preferences -> Intel Default Settings.
   If a Performance/Extreme choice is offered, select Performance. Leave
   Ai Overclock Tuner on Auto (XMP disabled). Leave voltage and CPU ratios at
   defaults; do not enable AI overclocking or load an old saved tuning profile.
9. Check fan/pump settings and that Ubuntu on the WD_BLACK NVMe is the boot
   target. Compare storage/VMD and Secure Boot settings with your photographs;
   do not guess at unfamiliar settings. If necessary, ask before saving them.
10. Press F10, review the changes, and save to boot Ubuntu. Leave the USB in
    until flashing and automatic restarts have finished. If a boot menu appears,
    choose Ubuntu, not the USB.
11. Return to this conversation for post-update checks before running training.

## If something goes wrong

- BIOS works but Ubuntu does not boot: inspect boot order and changed storage/
  Secure Boot settings first. Do not reinstall Ubuntu or format the internal SSD.
- No normal BIOS/boot after a failed update: this exact board supports rear-button
  USB BIOS FlashBack. A5459.CAP is already prepared on the USB. That recovery
  requires the designated FlashBack USB port, the PC shut down with PSU power
  present, and ASUS's specific button/LED procedure. Follow the official guide
  below with assistance; do not improvise or start it during an active EZ Flash.
- FlashBack is a recovery option, not a recovery guarantee; service may be needed
  if recovery fails. Clear CMOS resets settings but does not rewrite damaged BIOS.
- Flash completes but crashes continue: firmware cannot establish or restore CPU
  hardware health. We will test at defaults and isolate CPU/RAM/software causes.

## Recorded baseline and firmware provenance

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

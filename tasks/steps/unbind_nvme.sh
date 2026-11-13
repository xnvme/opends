#!/bin/bash
# SPDX-License-Identifier: BSD-3-Clause
# Unmount and hand an NVMe device to userspace (xnvme upcie-cuda), leaving
# the controller in a clean, verified power-on state.
#
# HOMI_NVME_DRIVER  uio_pci_generic or vfio-pci; unset picks by the IOMMU.
set -e

if [ $# -ne 2 ]; then
	echo "usage: unbind_nvme.sh BDF MOUNT" >&2
	exit 1
fi

BDF=$1
MOUNT=$2
HERE=$(dirname "$0")

if mountpoint -q "$MOUNT"; then
	echo "unmounting $MOUNT"
	umount "$MOUNT"
fi

# Re-enumerate the function before unbinding: a PCI remove + rescan forces the
# kernel nvme probe (reset, wait for CSTS.RDY, identify), leaving the
# controller in a verified-live state before handing it to userspace.
echo "re-enumerating $BDF"
echo 1 > "/sys/bus/pci/devices/$BDF/remove"
echo 1 > /sys/bus/pci/rescan

NS=""
for _ in $(seq 1 50); do
	if [ -d "/sys/bus/pci/devices/$BDF/nvme" ]; then
		if NS=$("$HERE/resolve_nvme_ns.sh" "$BDF" 2>/dev/null) &&
		   [ -b "$NS" ]; then
			break
		fi
	fi
	NS=""
	sleep 0.2
done
if [ -z "$NS" ]; then
	echo "error: $BDF did not come back as a live nvme namespace after rescan" >&2
	exit 1
fi
echo "$BDF live as $NS"

echo "unbinding $BDF from nvme"
echo "$BDF" > /sys/bus/pci/drivers/nvme/unbind

if [ -e "/sys/bus/pci/devices/$BDF/driver" ]; then
	DRIVER=$(basename "$(readlink "/sys/bus/pci/devices/$BDF/driver")")
	echo "error: $BDF still bound to $DRIVER after unbind"
	exit 1
fi
echo "$BDF unbound"

# Unbinding shuts the controller down (CC.SHN). Follow with an FLR to return
# it to a clean power-on state xnvme's open can enable, then settle so the
# open does not race the controller-level reset.
echo "resetting $BDF"
echo 1 > "/sys/bus/pci/devices/$BDF/reset"
sleep 1

# xNVMe takes the drive through vfio-pci behind a translating IOMMU and through
# uio_pci_generic without one. HOMI_NVME_DRIVER names the driver instead.
# driver_override pins the binding to this device only.
USER_DRIVER=${HOMI_NVME_DRIVER:-}
if [ -z "$USER_DRIVER" ]; then
	USER_DRIVER=uio_pci_generic
	case "$(cat "/sys/bus/pci/devices/$BDF/iommu_group/type" 2>/dev/null)" in
		DMA | DMA-FQ) USER_DRIVER=vfio-pci ;;
	esac
fi
case "$USER_DRIVER" in
	uio_pci_generic | vfio-pci) ;;
	*)
		echo "error: HOMI_NVME_DRIVER is '$USER_DRIVER'; use uio_pci_generic or vfio-pci" >&2
		exit 1
		;;
esac
modprobe "$USER_DRIVER"
echo "$USER_DRIVER" > "/sys/bus/pci/devices/$BDF/driver_override"
echo "$BDF" > "/sys/bus/pci/drivers/$USER_DRIVER/bind"
DRIVER=$(basename "$(readlink "/sys/bus/pci/devices/$BDF/driver")" 2>/dev/null)
if [ "$DRIVER" != "$USER_DRIVER" ]; then
	echo "error: $BDF not bound to $USER_DRIVER (got '$DRIVER')" >&2
	exit 1
fi
echo "$BDF bound to $USER_DRIVER"

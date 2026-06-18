# Missing V100 GPU Analysis

## Issue

On some boots, one V100 GPU failed to appear even though the hardware was
physically present. The missing GPU could change between boots, which pointed
away from a single permanently failed module.

## Finding

The evidence supported a PCIe link-training timing issue:

- the switch port detected physical presence;
- the communication link did not train;
- the sibling GPU on the same board could work normally;
- software rescans did not revive the dead link;
- a full cold power cycle was the likely immediate recovery path.

## Why Software Recovery Was Not Enough

Linux can rescan the bus, wake bridge devices, or ask a link to retrain, but it
cannot fully power-cycle an SXM2 module if the module is stuck before link
training completes. A warm reboot may also leave standby power in place, which
means the failed state can persist.

## Recommended Fixes

1. Use a cold chassis power cycle to reset the baseboard and switch state.
2. Check BIOS settings for PCIe link-training timeout or initialization delay.
3. Force Gen3 where the BIOS exposes the setting.
4. If the issue recurs, consider CMOS/NVRAM reset because stale PCIe allocation
   state can produce disappearing-GPU symptoms.

## Public Lesson

V100 fleet reliability is not only a CUDA question. Link training, PLX switch
behavior, cold power cycling, and BIOS settings can directly affect model
availability.

## Links

- https://proprietarylegal.ai
- https://proprietarylegal.com

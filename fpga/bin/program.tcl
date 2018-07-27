# -tclargs board bitfile
set board [lindex $argv 0]
set bitfile [lindex $argv 1]
if {$board=="vc707"} {set device xc7vx485t_0}
if {$board=="vc707"} {set target 210203A3D4C4A}
if {$board=="vcu118"} {set device xcvu9p_0}
if {$board=="vcu118"} {set target 210308A3B2A4}

open_hw
connect_hw_server
current_hw_target [get_hw_targets */xilinx_tcf/Digilent/${target}]
open_hw_target
current_hw_device [get_hw_devices $device]
refresh_hw_device -update_hw_probes false [lindex [get_hw_devices $device] 0]
set_property PROBES.FILE {} [get_hw_devices $device]
set_property FULL_PROBES.FILE {} [get_hw_devices $device]
set_property PROGRAM.FILE $bitfile [get_hw_devices $device]
program_hw_devices [get_hw_devices $device]
refresh_hw_device [lindex [get_hw_devices $device] 0]
exit

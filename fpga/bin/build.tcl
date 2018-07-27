# -tclargs board clkdiv init gridx gridy gridz
set board [lindex $argv 0]
set clkdiv [lindex $argv 1]
set init [lindex $argv 2]
set gridx [lindex $argv 3]
set gridy [lindex $argv 4]
set gridz [lindex $argv 5]

# set PERIOD parameter for 115200 baud
if {$board=="vc707" && $clkdiv=="div2"} {set period 434}
if {$board=="vc707" && $clkdiv=="div4"} {set period 217}
if {$board=="vcu118" && $clkdiv=="div4"} {set period 271}

# read verilog source
if {$clkdiv=="div2"} { read_verilog ../src/glue_div2.v }
if {$clkdiv=="div3"} { read_verilog ../src/glue_div3.v }
if {$clkdiv=="div4"} { read_verilog ../src/glue_div4.v }
read_verilog ../src/rng.v
read_verilog ../src/uart.v
read_verilog ../src/life.v
read_verilog ../src/ram.v
read_verilog ../src/core.v
read_verilog ../src/top.v

# synthesis
if {$board=="vc707"} {set part "xc7vx485tffg1761-2"}
if {$board=="vcu118"} {set part "xcvu9p-flga2104-2-e"}

set_param general.maxThreads 8
set_param synth.elaboration.rodinMoreOptions "rt::set_parameter max_loop_limit 200000"
set_param project.hsv.draftModeDefault only
set_property source_mgmt_mode DisplayOnly [current_project]

synth_design -top top -part $part -generic INIT=$init -generic GRIDX=$gridx -generic GRIDY=$gridy -generic GRIDZ=$gridz -generic PERIOD=$period

set_property ALLOW_COMBINATORIAL_LOOPS TRUE [get_nets rng_inst/*]
set_property SEVERITY {Warning}  [get_drc_checks LUTLP-1]
set_property SEVERITY {Warning} [get_drc_checks NSTD-1]

if {$board=="vc707"} {
    create_clock -period 5.0 [get_ports SYSCLK_P]
    set_property PACKAGE_PIN E19 [get_ports SYSCLK_P]
    set_property IOSTANDARD LVDS [get_ports SYSCLK_P]
    set_property PACKAGE_PIN E18 [get_ports SYSCLK_N]
    set_property IOSTANDARD LVDS [get_ports SYSCLK_N]
    set_property PACKAGE_PIN AU36 [get_ports USB_UART_TX]
    set_property IOSTANDARD LVCMOS18 [get_ports USB_UART_TX]
    set_property PACKAGE_PIN AU33 [get_ports USB_UART_RX]
    set_property IOSTANDARD LVCMOS18 [get_ports USB_UART_RX]
}
if {$board=="vcu118"} {
    create_clock -period 4.0 [get_ports SYSCLK_P]
    set_property PACKAGE_PIN E12 [get_ports SYSCLK_P]
    set_property IOSTANDARD LVDS [get_ports SYSCLK_P]
    set_property PACKAGE_PIN D12 [get_ports SYSCLK_N]
    set_property IOSTANDARD LVDS [get_ports SYSCLK_N]
    set_property PACKAGE_PIN BB21 [get_ports USB_UART_TX]
    set_property IOSTANDARD LVCMOS18 [get_ports USB_UART_TX]
    set_property PACKAGE_PIN AW25 [get_ports USB_UART_RX]
    set_property IOSTANDARD LVCMOS18 [get_ports USB_UART_RX]
}

# p&r
opt_design
place_design
phys_opt_design
route_design
report_route_status
report_timing_summary
report_timing -max_paths 100 -unique_pins -path_type full
report_utilization

set_property BITSTREAM.GENERAL.COMPRESS TRUE [current_design]
write_bitstream -force ${board}_${clkdiv}_${init}_${gridx}_${gridy}_${gridz}.bit

exit

// top level
module glue (SYSCLK_P,SYSCLK_N,USB_UART_RX,USB_UART_TX,clk,reset,rx,tx);
input SYSCLK_P;
input SYSCLK_N;
input USB_UART_RX;
output USB_UART_TX;
output clk;
output reset;
output rx;
input tx;

wire clkr,sysclk,EOS;

IBUFGDS #(
.DIFF_TERM("FALSE"), // Differential Termination
.IBUF_LOW_PWR("FALSE"), // Low power="TRUE", Highest performance="FALSE"
.IOSTANDARD("DEFAULT") // Specify the input I/O standard
) IBUFGDS_inst (
    .O(sysclk), // Clock buffer output
    .I(SYSCLK_P), // Diff_p clock buffer input (connect directly to top-level port)
    .IB(SYSCLK_N) // Diff_n clock buffer input (connect directly to top-level port)
);

BUFR #(
.BUFR_DIVIDE("3"), // Values: "BYPASS, 1, 2, 3, 4, 5, 6, 7, 8"
.SIM_DEVICE("7SERIES") // Must be set to "7SERIES"
)
BUFR_inst (
    .O(clkr), // 1-bit output: Clock output port
    .CE(1'b1), // 1-bit input: Active high, clock enable input
    .CLR(1'b0), // 1-bit input: ACtive high reset input
    .I(sysclk) // 1-bit input: Clock buffer input driven by an IBUFG, MMCM or local interconnect
);

BUFG BUFG_inst (
    .O(clk), // 1-bit output: Clock output
    .I(clkr) // 1-bit input: Clock input
);

IBUF #(
.IBUF_LOW_PWR("TRUE"),  // Low power (TRUE) vs. performance (FALSE) setting for referenced I/O standards
.IOSTANDARD("DEFAULT")  // Specify the input I/O standard
) IBUF_inst (
    .O(rx),     // Buffer output
    .I(USB_UART_RX)      // Buffer input (connect directly to top-level port)
);

OBUF #(
.DRIVE(12),   // Specify the output drive strength
.IOSTANDARD("DEFAULT"), // Specify the output I/O standard
.SLEW("SLOW") // Specify the output slew rate
) OBUF_inst (
    .O(USB_UART_TX),     // Buffer output (connect directly to top-level port)
    .I(tx)      // Buffer input
);

STARTUPE2 #(
.PROG_USR("FALSE"),  // Activate program event security feature. Requires encrypted bitstreams.
.SIM_CCLK_FREQ(0.0)  // Set the Configuration Clock Frequency(ns) for simulation.
)
STARTUPE2_inst (
    .CFGCLK(),       // 1-bit output: Configuration main clock output
    .CFGMCLK(),     // 1-bit output: Configuration internal oscillator clock output
    .EOS(EOS),             // 1-bit output: Active high output signal indicating the End Of Startup.
    .PREQ(),           // 1-bit output: PROGRAM request to fabric output
    .CLK(1'b0),             // 1-bit input: User start-up clock input
    .GSR(1'b0),             // 1-bit input: Global Set/Reset input (GSR cannot be used for the port name)
    .GTS(1'b0),             // 1-bit input: Global 3-state input (GTS cannot be used for the port name)
    .KEYCLEARB(1'b0), // 1-bit input: Clear AES Decrypter Key input from Battery-Backed RAM (BBRAM)
    .PACK(1'b0),           // 1-bit input: PROGRAM acknowledge input
    .USRCCLKO(1'b0),   // 1-bit input: User CCLK input
    .USRCCLKTS(1'b1), // 1-bit input: User CCLK 3-state enable input
    .USRDONEO(1'b1),   // 1-bit input: User DONE pin output control
    .USRDONETS(1'b1)  // 1-bit input: User DONE 3-state enable output
);

reg [7:0] resetcnt;
reg resetp;
always @ (posedge clk or negedge EOS)
begin
    if (!EOS) begin
        resetcnt='d0;
        resetp='b1;
    end
    else if (resetcnt=='d100)
        resetp='b0;
    else
        resetcnt=resetcnt+'d1;
end

BUFG BUFG_reset (
    .O(reset), // 1-bit output: Clock output
    .I(resetp)  // 1-bit input: Clock input
);
endmodule

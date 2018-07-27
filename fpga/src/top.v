// top level
module top (
SYSCLK_P,
SYSCLK_N,
USB_UART_RX,
USB_UART_TX
);
input SYSCLK_P;
input SYSCLK_N;
input USB_UART_RX;
output USB_UART_TX;

// parameters
parameter INIT=20; // master parameter
parameter GRIDX=600; // master parameter
parameter GRIDY=40; // master parameter
parameter GRIDZ=15; // master parameter
parameter PERIOD=434; // master parameter

// top level io,clock,reset circuit
wire clk,reset,rx,tx;
glue glue_inst(.SYSCLK_P(SYSCLK_P),.SYSCLK_N(SYSCLK_N),.USB_UART_RX(USB_UART_RX),.USB_UART_TX(USB_UART_TX),.clk(clk),.reset(reset),.rx(rx),.tx(tx));

// uart interface
wire break,run;
wire life;
wire [1:0] rng_mode;
wire [31:0] num_init; // number of random initializations to try
wire [INIT*INIT+32-1:0] life_data;
uart_tx #(INIT,PERIOD) uart_tx_inst(.clk(clk),.reset(reset),.break(break),.tx(tx),.life(life),.halt(1'b0),.data(life_data));
uart_rx #(PERIOD) uart_rx_inst(.clk(clk),.reset(reset),.rx(rx),.break(break),.run(run),.data({rng_mode,num_init}));

// random number generator
wire [(INIT*INIT)-1:0] rng_data;
rng #(INIT) rng_inst (clk,rng_mode,rng_data,reset|break);

// core logic
core #(INIT,GRIDX,GRIDY,GRIDZ) core_inst (clk,reset|break,run,life,num_init,rng_data,life_data);

endmodule

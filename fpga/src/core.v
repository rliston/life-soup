// core logic
module core (clk,reset,run,life,num_init,rng_data,life_data);
parameter INIT=20; // init size
parameter GRIDX=1500; // grid width
parameter GRIDY=100; // grid height = GRIDY*GRIDZ
parameter GRIDZ=15; // grid height = GRIDY*GRIDZ
input clk;
input reset;
input run;
output life;
input [31:0] num_init;
input [(INIT*INIT)-1:0] rng_data;
output [INIT*INIT+32-1:0] life_data;
reg life;
reg [INIT*INIT+32-1:0] life_data;

// datapath
genvar i,j;
integer k,n,m;
reg [4:0] a;

// init mux
reg init;
reg [INIT*INIT-1:0] rng_init;
reg [INIT*INIT-1:0] best_rng;
reg [GRIDX*GRIDY-1:0] initd;
// MIDDLE centers the init patch in the middle GRIDZ slice, GRIDZ should be an odd number
// STRADDLE centers the init patch across the middle two GRIDZ slices, GRIDZ should be an even number
`ifdef MIDDLE
always @(a or rng_init) begin
    initd = 'b0;
    if (a == (GRIDZ/2)) begin
        for (n=0; n < INIT; n=n+1) begin // y
            for (m=0; m < INIT; m=m+1) begin // x
                initd[((GRIDY/2-INIT/2)+n)*GRIDX + ((GRIDX/2-INIT/2)+m)] = rng_init[n*INIT+m];
            end
        end
    end
end
`endif
`ifdef STRADDLE
always @(a or rng_init) begin
    initd = 'b0;
    if (a == ((GRIDZ/2)-1)) begin // bottom half
        for (n=0; n < INIT/2; n=n+1) begin // y
            for (m=0; m < INIT; m=m+1) begin // x
                initd[((GRIDY-INIT/2)+n)*GRIDX + ((GRIDX/2-INIT/2)+m)] = rng_init[n*INIT+m];
            end
        end
    end
    if (a == (GRIDZ/2)) begin // top half
        for (n=0; n < INIT/2; n=n+1) begin // y
            for (m=0; m < INIT; m=m+1) begin // x
                initd[n*GRIDX + ((GRIDX/2-INIT/2)+m)] = rng_init[(n+INIT/2)*INIT+m];
            end
        end
    end
end
`endif

// main grid
wire [GRIDX*GRIDY-1:0] d;
wire [GRIDX*GRIDY-1:0] spo;
ramsp #(GRIDX*GRIDY) grid_inst (clk, 1'b1, a, d, spo);

// double buffered storage for previous step top and bottom rows, write to the back buffer, read from the front buffer
reg [31:0] step_count;
wire back = step_count[0];
wire [GRIDX-1:0] dtop0;
wire [GRIDX-1:0] dtop1;
wire [GRIDX-1:0] dbot0;
wire [GRIDX-1:0] dbot1;
ramsp #(GRIDX) top0 (clk, init | !back, !back ? a : a-5'd1, d[GRIDX*GRIDY-1:GRIDX*GRIDY-GRIDX], dtop0);
ramsp #(GRIDX) top1 (clk, init | back, back ? a : a-5'd1, d[GRIDX*GRIDY-1:GRIDX*GRIDY-GRIDX], dtop1);
ramsp #(GRIDX) bot0 (clk, init | !back, !back ? a : a+5'd1, d[GRIDX-1:0], dbot0);
ramsp #(GRIDX) bot1 (clk, init | back, back ? a : a+5'd1, d[GRIDX-1:0], dbot1);
wire [GRIDX-1:0] prevtop;
wire [GRIDX-1:0] nextbot;
assign prevtop = (a=='d0) ? 'b0 : (back ? dtop0 : dtop1);
assign nextbot = (a==(GRIDZ-1)) ? 'b0 : (back ? dbot0 : dbot1);

wire [GRIDX*GRIDY-1:0] lifed;
assign d = init ? initd : lifed;

// life
`define BOTTOM ((i/GRIDX)==0)
`define TOP ((i/GRIDX)==(GRIDY-1))
`define LEFT ((i%GRIDX)==0)
`define RIGHT ((i%GRIDX)==(GRIDX-1))
for (i=0; i<GRIDX*GRIDY; i=i+1) begin
    life life_inst (
    .out(lifed[i]),
    .c(spo[i]),
    .n(`TOP ? nextbot[i%GRIDX] : spo[i+GRIDX]),
    .s(`BOTTOM ? prevtop[i%GRIDX] : spo[i-GRIDX]),
    .e(`RIGHT ? 1'b0 : spo[i+1]),
    .w(`LEFT ? 1'b0 : spo[i-1]),
    .nw(`LEFT ? 1'b0 : (`TOP ? nextbot[(i%GRIDX)-1] : spo[i+GRIDX-1])),
    .ne(`RIGHT ? 1'b0 : (`TOP ? nextbot[(i%GRIDX)+1] : spo[i+GRIDX+1])),
    .sw(`LEFT ? 1'b0 : (`BOTTOM ? prevtop[(i%GRIDX)-1] : spo[i-GRIDX-1])),
    .se(`RIGHT ? 1'b0 : (`BOTTOM ? prevtop[(i%GRIDX)+1] : spo[i-GRIDX+1]))
    );
end

// t1,t2,ts checks
reg t1_we;
reg t2_we;
reg ts_we;
wire [GRIDX*GRIDY-1:0] t1_o;
wire [GRIDX*GRIDY-1:0] t2_o;
wire [GRIDX*GRIDY-1:0] ts_o;
wire [GRIDX*GRIDY-1:0] txd;
assign txd = init ? 'b0 : spo;
ramsp #(GRIDX*GRIDY) t1_inst (clk, t1_we, a, txd, t1_o);
ramsp #(GRIDX*GRIDY) t2_inst (clk, t2_we, a, txd, t2_o);
ramsp #(GRIDX*GRIDY) ts_inst (clk, ts_we, a, txd, ts_o);

reg [GRIDZ-1:0] t1_check;
reg [GRIDZ-1:0] t2_check;
reg [GRIDZ-1:0] ts_check;
always @(posedge clk) begin
    t1_check[a] <= (spo==t1_o);
    t2_check[a] <= (spo==t2_o);
    ts_check[a] <= (spo==ts_o);
end

// control fsm
reg [3:0] state;
reg [31:0] best_step;
reg [31:0] init_count;

always @(posedge clk) begin
    t1_we <= (step_count % 1)=='d0;
    t2_we <= (step_count % 2)=='d0;
    ts_we <= (step_count % 200)=='d0;
end

parameter IDLE = 'd1;
parameter INITIALIZE = 'd2;
parameter STEP = 'd3;
parameter CHECK = 'd4;

always @(posedge clk) begin
    if (reset)
        state <= IDLE;
    else
    case(state)
    IDLE : begin
        a <= 'd0;
        step_count <= 'b0;
        init_count <= 'b0;
        life <= 1'b0;
        init <= 1'b0;
        if (run) begin
            best_step <= 'd0;
            init <= 1'b1;
            rng_init <= rng_data;
            state <= INITIALIZE;
        end
    end
    INITIALIZE : begin
        a <= a+'d1;
        if (init_count >= num_init) begin
            life <= 1'b1;
            life_data <= {best_step,best_rng};
            state <= IDLE;
        end
        else if (a==(GRIDZ-1)) begin
            init <= 1'b0;
            step_count <= 'd0;
            init_count <= init_count+'d1;
            state <= CHECK;
        end
    end
    STEP : begin
        a <= a+'d1;
        if (a==(GRIDZ-1)) begin
            step_count <= step_count+'d1;
            state <= CHECK;
        end
    end
    CHECK : begin
        a <= 'd0;
        if (step_count > best_step) begin
            best_step <= step_count;
            best_rng <= rng_init;
        end
        if ((step_count != 'd0) && (t1_check=={GRIDZ{1'b1}} || t2_check=={GRIDZ{1'b1}} || ts_check=={GRIDZ{1'b1}} || (step_count > 'd100000))) begin
            init <= 1'b1;
            rng_init <= rng_data;
            state <= INITIALIZE;
        end
        else begin
            state <= STEP;
        end
    end
    endcase
end
endmodule

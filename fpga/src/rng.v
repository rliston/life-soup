module rng_ring (clk,x);
parameter SIZE=400;
input clk;
output [SIZE-1:0] x;

genvar i;
wire [SIZE-1:0] i0,i1,i2,o;
for(i=0; i<SIZE; i=i+1) begin
    if (i==0)
        XNOR3 xnor_inst (.O (o[0]), .I0 (o[SIZE-1]), .I1 (o[0]), .I2 (o[1]));
    else if (i==SIZE-1)
        XOR3 xor_inst (.O (o[SIZE-1]), .I0 (o[SIZE-2]), .I1 (o[SIZE-1]), .I2 (o[0]));
    else
        XOR3 xor_inst (.O (o[i]), .I0 (o[i-1]), .I1 (o[i]), .I2 (o[i+1]));
end

reg [SIZE-1:0] x, q;
always @(posedge clk) begin
    q <= o;
    x <= q;
end
endmodule

module rng(clk,cfg,x,reset);
parameter INIT=20;
localparam NI=INIT*INIT;
input clk;
input [1:0] cfg;
output [NI-1:0] x;
input reset;
reg [NI-1:0] x;

wire [NI-1:0] x0;
rng_ring #(NI) rng_0 (clk,x0);
wire [NI-1:0] x1;
rng_ring #(NI) rng_1 (clk,x1);
wire [NI-1:0] x2;
rng_ring #(NI) rng_2 (clk,x2);
wire [NI-1:0] x3;
rng_ring #(NI) rng_3 (clk,x3);

reg [2:0] mc;
reg [2:0] rc;
always @(posedge clk) begin
    if (reset) begin
        mc <= 3'b000;
        rc <= 3'b000;
    end
    else begin
        if (x0[0])
            mc <= mc + 3'b001;
        if (cfg==2'b00)
            rc <= 3'b111;
        if (cfg==2'b01)
            rc <= 3'b001;
        if (cfg==2'b10)
            rc <= 3'b010;
        if (cfg==2'b11)
            rc <= mc;
    end
end

always @(posedge clk) begin
    if (rc==3'b000)
        x <= x0^x1;
    else if (rc==3'b001)
        x <= x0&x1&x2&x3;
    else if (rc==3'b010)
        x <= x0&x1&x2;
    else if (rc==3'b011)
        x <= x0&x1&x2;
    else if (rc==3'b100)
        x <= x0&x1&x2&x3;
    else if (rc==3'b101)
        x <= x0|x1|x2;
    else if (rc==3'b110)
        x <= x0|x1|x2|x3;
    else if (rc==3'b111)
        x <= x0^x1^x2^x3;
end
endmodule

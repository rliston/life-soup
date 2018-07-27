// RAM with Asynchronous Read (Distributed RAM) 32 deep

module ramsp (clk, we, a, di, spo);
parameter WIDTH=1;
input clk;
input we;
input [4:0] a;
input [WIDTH-1:0] di;
output [WIDTH-1:0] spo;

reg [WIDTH-1:0] ram [31:0];
always @(posedge clk) 
begin 
 if (we)
    ram[a] <= di;
end
assign spo = ram[a];

endmodule

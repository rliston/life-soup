module testbench;
// parameters
parameter INIT=20; // master parameter
//parameter GRIDX=1500; // master parameter
//parameter GRIDY=100; // master parameter
parameter GRIDX=600; // master parameter
parameter GRIDY=40; // master parameter
//parameter GRIDX=375; // master parameter
//parameter GRIDY=25; // master parameter

reg clk,reset,run;
reg [31:0] num_init;
reg [31:0] act_thresh;
reg [(INIT*INIT)-1:0] rng_data;
wire life;
wire [INIT*INIT+32+32-1:0] life_data;

// dut
core #(INIT,GRIDX,GRIDY) core_inst (clk,reset,run,life,num_init,act_thresh,rng_data,life_data);

reg [(INIT*INIT)-1:0] mem[1:0];

initial begin
// -testplusarg ZEROINIT
if ($test$plusargs("ZEROINIT")) begin
    $display("ZEROINIT");
    assign num_init = 'd10;
    assign act_thresh = 'd100000000;
    assign rng_data = 400'b0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000;
end

// -testplusarg FASTDIE
if ($test$plusargs("FASTDIE")) begin
    $display("FASTDIE");
    assign num_init = 'd1;
    assign act_thresh = 'd100000000;
    assign rng_data = 400'b1000001010000001101110110100110111000111000110000010010010011110100011000010010110011101111010000110111001100100011011111100001000100110000101101010000010001000011000111100000001100000101000001100111011001101010000001011110100011011001000110000000011111000011100000100010000111111001011111100001110110111100000100000010110101100100011001001100110111011001001000110011000000101011000100001010100111110; // 23 steps
    //assign rng_data = 400'b0010011110010101100010000110010101000110011100110000011101111010010100000110011111111011110111000101001001100111101011010111100010001101010010001100010111010101001010100111101111100101001000001101100100000110011000110001111000100000000111011000011001011100001101001010100000111011001000111100111110010100110101010001100101001010001001100000100001000111101000011111110110111011100111111011110101110001; // 44 steps
    //assign rng_data = 400'b0110100010101001000001010110101010011010111110001011110010101000110001000111101100100101000111111001111000110010011000010001110111010110110110101111010011001011010101011011001010111000101001100000010111001001001001001111111011110101001111110111111010011110001000101001011111000110111111000111001100001111011111100000011010110110000111010001000101000100011101100101011011111011001011111000000100101000; // 179 steps
end

// -testplusarg GLIDERDIE
if ($test$plusargs("GLIDERDIE")) begin
    $display("GLIDERDIE");
    assign num_init = 'd1;
    assign act_thresh = 'd0;
    assign rng_data = 400'b1001100110011111011101110100001010111000000111110011111110100001010010111001110100001110100111101101111010100110110111101001110110001100000111101011111001010101111001110000110100011111111010101100110001011011001100101110101100111000001101001111100111101100011010101111111111011000111010010001001100100010100100110100110001000010000010001000001001111100011011101001011000011111001001100001011001100110; // 759 steps
end

// -testplusarg FILEINIT
if ($test$plusargs("FILEINIT")) begin
    $display("FILEINIT");
    assign num_init = 'd1;
    assign act_thresh = 'd1000000;
    $readmemb("../test/current.txt", mem);
    assign rng_data = mem[0];
end

end

// test
always #5  clk =  !clk; 

integer f;
initial begin
    f = $fopen("output.txt","w");
    clk=0;
    reset=1;
    run=0;
    #100 reset=0;
    #100 run=1;
    #10 run=0;
    #10000000 begin
        $display("TIMEOUT");
        $finish; // failsafe
    end
end

wire [INIT*INIT-1:0] rng_init;
wire [31:0] step_count;
wire [31:0] boundact;
assign {boundact,step_count,rng_init} = life_data;
always @ (posedge clk) begin
    if (life==1'b1) begin
        $displayb("LIFE boundact %d step_count %d rng_init %b",boundact,step_count,rng_init);
        #100 $finish;
    end
end

// monitor
parameter IDLE = 'd1;
parameter INITIALIZE = 'd2;
parameter STEP = 'd3;
parameter CHECK = 'd4;
integer i,j,k;
always @(posedge clk) begin
    //$display("state %d",core_inst.state);
    if (core_inst.state == CHECK) begin
        $fwrite(f,"check\n");
        for (i=0;i<15;i=i+1) begin
            for (j=0;j<GRIDY;j=j+1) begin
                for (k=0;k<GRIDX;k=k+1) begin
                    $fwrite(f,"%b",core_inst.grid_inst.ram[i][j*GRIDX+k]);
                end
            $fwrite(f,"\n");
            end
        end
    end
end

always @(core_inst.step_count)
    $display("step_count %d boundact %d",core_inst.step_count,core_inst.boundact);

endmodule

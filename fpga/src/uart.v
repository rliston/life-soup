// uart tx : assert halt for 1 clock, or life and data for 1 clock
module uart_tx (clk,reset,break,tx,life,halt,data);
parameter INIT=20;
parameter PERIOD=434; // 100MHz clk, 115200 baud
localparam NI=INIT*INIT;
input clk;
input reset;
input break;
output tx;
input life;
input halt;
input [NI+32-1:0] data;

parameter IDLE = 'd1;
parameter HALT = 'd2;
parameter LIFE = 'd3;
parameter DATA = 'd4;
parameter SEND = 'd5;

reg tx;
reg [3:0] state;
reg [3:0] return;
reg [7:0] char;
reg [13:0] scnt;
reg [15:0] dcnt;

always @(posedge clk) begin
    if (reset || break) begin
        state <= IDLE;
    end
    else
    case(state)
    IDLE : begin
        scnt <= 'd0;
        tx <= 'b1;
        if (life)
            state <= LIFE;
        else if (halt)
            state <= HALT;
    end
    HALT : begin
        char <= 'h48;
        return <= IDLE;
        state <= SEND;
    end
    LIFE : begin
        char <= 'h4c;
        return <= DATA;
        dcnt <= 'd0;
        state <= SEND;
    end
    DATA : begin
        if (dcnt==NI+32)
            state <= IDLE;
        else begin
            char <= data[dcnt] ? 'h31 : 'h30;
            dcnt <= dcnt+'d1;
            return <= DATA;
            state <= SEND;
        end
    end
    SEND : begin
        case(scnt)
        0 : tx <= 'b0; // start bit
        (PERIOD*2)*1 : tx <= char[0];
        (PERIOD*2)*2 : tx <= char[1];
        (PERIOD*2)*3 : tx <= char[2];
        (PERIOD*2)*4 : tx <= char[3];
        (PERIOD*2)*5 : tx <= char[4];
        (PERIOD*2)*6 : tx <= char[5];
        (PERIOD*2)*7 : tx <= char[6];
        (PERIOD*2)*8 : tx <= char[7];
        (PERIOD*2)*9 : tx <= 'b1; // stop bit
        endcase
        if (scnt==(PERIOD*2)*10) begin
            scnt <= 'd0;
            state <= return;
        end
        else
            scnt <= scnt+'d1;
    end
    endcase
end
endmodule

// uart rx : assert break for 1 clock, or run and data for 1 clock
module uart_rx (clk,reset,rx,break,run,data);
parameter PERIOD=434; // 100MHz clk, 115200 baud
input clk;
input reset;
input rx;
output break;
output run;
output [(32+2)-1:0] data;

parameter IDLE = 'd1;
parameter CMD = 'd2;
parameter DATA = 'd3;
parameter RECV = 'd4;
parameter WAIT = 'd5;
reg [3:0] state;
reg [3:0] return;
reg [7:0] char;
reg [13:0] rcnt;
reg [7:0] dcnt;
reg [(32+2)-1:0] data;
reg break;
reg run;
reg [7:0] cmd;

always @(posedge clk) begin
    if (reset) begin
        state <= IDLE;
    end
    else
    case(state)
    IDLE : begin
        break <= 'b0;
        run <= 'b0;
        return <= CMD;
        state <= WAIT;
    end
    CMD : begin
        if ((char=='h52) || (char=='h41) || (char=='h4d)) begin // 'R','A','M'
            cmd <= char;
            dcnt <= 'd0;
            return <= DATA;
            state <= WAIT;
        end
        else begin
            break <= 'b1; // debug
            state <= IDLE; // unknown command
        end
    end
    DATA : begin
        data[dcnt] <= (char=='h30) ? 'b0 : 'b1;
        if (dcnt=='d33) begin
            if (cmd=='h52)
                run <= 'b1;
            state <= IDLE;
        end
        else begin
            dcnt <= dcnt+'d1;
            return <= DATA;
            state <= WAIT;
        end
    end
    WAIT : begin
        rcnt <= 'd0;
        if (!rx)
            state <= RECV;
    end
    RECV : begin
        case(rcnt)
        0+PERIOD :
            if (rx) begin
                break <= 'b1;
                state <= IDLE; // start bit should be zero
            end
        (PERIOD*2)*1+PERIOD : char[0] <= rx;
        (PERIOD*2)*2+PERIOD : char[1] <= rx;
        (PERIOD*2)*3+PERIOD : char[2] <= rx;
        (PERIOD*2)*4+PERIOD : char[3] <= rx;
        (PERIOD*2)*5+PERIOD : char[4] <= rx;
        (PERIOD*2)*6+PERIOD : char[5] <= rx;
        (PERIOD*2)*7+PERIOD : char[6] <= rx;
        (PERIOD*2)*8+PERIOD : char[7] <= rx;
        (PERIOD*2)*9+PERIOD : // stop bit
            if (!rx) begin // stop bit should be one, else break
                break <= 'b1;
                state <= IDLE;
            end
        endcase
        if (rcnt==(PERIOD*2)*10)
            state <= return;
        else
            rcnt <= rcnt+'d1;
    end
    endcase
end
endmodule

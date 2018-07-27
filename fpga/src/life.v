// life
module life (out,c,n,s,e,w,nw,ne,sw,se);
input c,n,s,e,w,nw,ne,sw,se;
output out;
wire [3:0] sum;
assign sum = n+s+e+w+nw+ne+sw+se;
assign out = (c && (sum==2)) || (sum==3);
endmodule

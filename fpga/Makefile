default::
	echo "make {clean,sim,build,program,minicom}"
build::
	cd ./build ; vivado -stack 10000 -log build.log -mode tcl -source ../bin/build.tcl -tclargs vcu118 div4 20 1000 40 25 ;
program::
	cd ./build ; vivado -log program.log -mode tcl -source ../bin/program.tcl -tclargs vcu118 $(BIT) ;


# SIMULATION TARGETS
waveform::
	cd ./sim ; xsim --gui --view --wdb testbench.wdb
sim::
	cd ./sim ; xvlog ../src/testbench.v ../src/core.v ../src/life.v ../src/ram.v ;
	cd ./sim ; xelab testbench -timescale 1ns/1ps --debug typical
	cd ./sim ; xsim testbench --stats --testplusarg FILEINIT --log testbench.log --tclbatch ../bin/sim.tcl --wdb testbench.wdb
regression::
	cd ./sim ; python ../bin/genlif.py --regression
fileinit::
	cd ./sim ; xvlog ../src/testbench.v ../src/core.v ../src/life.v ../src/ram.v ;
	cd ./sim ; xelab testbench -timescale 1ns/1ps --debug typical
	cd ./sim ; xsim testbench --stats --testplusarg FILEINIT --runall --nolog
gliderdie::
	cd ./sim ; xvlog ../src/testbench.v ../src/core.v ../src/life.v ../src/ram.v ;
	cd ./sim ; xelab testbench -timescale 1ns/1ps --debug typical
	cd ./sim ; xsim testbench --stats --testplusarg GLIDERDIE --runall --nolog
fastdie::
	cd ./sim ; xvlog ../src/testbench.v ../src/core.v ../src/life.v ../src/ram.v ;
	cd ./sim ; xelab testbench -timescale 1ns/1ps --debug typical
	cd ./sim ; xsim testbench --stats --testplusarg FASTDIE --runall --nolog

minicom::
	minicom --device /dev/ttyUSB2 --baudrate 115200 --8bit
clean:
	rm -rf ./build/* ./sim/*

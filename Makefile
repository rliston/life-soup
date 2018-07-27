default:
	echo "make r01"
vc707:
	python ./bin/life.py --num_init 1000000 --nocheck --rngmode 00 --dir ./lif/r00/ --serial /dev/ttyUSB0
vcu118:
	python ./bin/life.py --num_init 1000000 --nocheck --rngmode 00 --dir ./lif/r00.2/ --serial /dev/ttyUSB4

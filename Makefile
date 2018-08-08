default:
	echo "make {i20,i50}"
i20:
	python ./bin/life.py --num_init 10000000 --nocheck --rngmode 00 --dir ./lif/r00i20/ --serial /dev/ttyUSB0
i50:
	python ./bin/life.py --num_init 10000000 --nocheck --rngmode 00 --dir ./lif/r00i50/ --serial /dev/ttyUSB4 --size 1200 --init 50
histogram:
	cd ./lif/histogram ; python ../../bin/histogram.py --vis

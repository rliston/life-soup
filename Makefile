default:
	echo "make {i20,i50}"
i20:
	# Silicon_Labs_CP2103_USB_to_UART_Bridge_Controller_0001
	python ./bin/life.py --num_init 100000000 --nocheck --rngmode 00 --dir ./lif/r00i20/ --serial /dev/ttyUSB5 --size 750 --init 20
i50:
	# Silicon_Labs_CP2105_Dual_USB_to_UART_Bridge_Controller_007F035C (second entry)
	python ./bin/life.py --num_init 20000000 --nocheck --rngmode 00 --dir ./lif/r00i50/ --serial /dev/ttyUSB4 --size 1200 --init 50
histogram:
	cd ./lif/histogram ; python ../../bin/histogram.py --vis

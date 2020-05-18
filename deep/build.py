# sudo build.py
import lifelib ; print('lifelib',lifelib.__version__)
sess = lifelib.load_rules("b3s23")
print(dir(sess))

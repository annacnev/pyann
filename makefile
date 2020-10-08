LIB_DIR = pyann/annlib/lib
ANNLIB_DIR = pyann/annlib

default: pyann

pyann: setup.py $(ANNLIB_DIR)/annlib.pyx $(LIB_DIR)/libNN.a


$(LIB_DIR)/libNN.a:
	make -C $(LIB_DIR) libNN.a

clean:
	rm *.so
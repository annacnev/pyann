
CC = g++

default: libNN.a

libNN.a: NN.o
	ar rcs $@ $^

NN.o: NN.cc
	$(CC) -c $<

clean:
	rm *.o *.a
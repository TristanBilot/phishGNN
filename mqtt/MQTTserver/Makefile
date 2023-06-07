all: mqtt_pub mqtt_sub
	echo "Build finish!"

mqtt_pub: clientTester.c
	gcc clientTester.c -o clientTester -lmosquitto

mqtt_sub: server.c
	gcc server.c -o server -lmosquitto

clean:
	rm server clientTester

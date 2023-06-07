#include <stdio.h>
#include <mosquitto.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// FOR TEST -> Do in javascript

void on_connect(struct mosquitto *mosq, void *obj, int rc){

  printf("Connected with id %d\n",* (int *) obj);
  if(rc){
    printf("error\n");
    exit(-1);
  }
  mosquitto_subscribe(mosq, NULL, "response/56", 0); // REPLACE 56 by random ID
  printf("subbed to topic 56\n");
}

void on_message(struct mosquitto *mosq, void *obj, const struct mosquitto_message *msg){

    printf("You got an answer : %s\n",(char *) msg->payload);

    }

int main(int argc, char *argv[]){
	int rc;
  
  int id=56;
	struct mosquitto * mosq;

	mosquitto_lib_init();

	mosq = mosquitto_new("client-test", true, &id);
  mosquitto_connect_callback_set(mosq, on_connect); // When connected to the broker
  mosquitto_message_callback_set(mosq, on_message); // When a message is received (responses)

	rc = mosquitto_connect(mosq, "localhost", 8080, 60);
	if(rc != 0){
		printf("Client could not connect to broker! Error Code: %d\n", rc);
		mosquitto_destroy(mosq);
		return -1;
	}
	printf("We are now connected to the broker!\n");
	mosquitto_publish(mosq, NULL, "request/56", strlen(argv[1]), argv[1] ,0, false); // Reaplace 56 by random ID
  printf("Published\n");
  mosquitto_loop_start(mosq);
  printf("Waiting for response\n");
  printf("Enter quit\n");
  getchar();
  mosquitto_loop_stop(mosq,true);

  mosquitto_disconnect(mosq);
  mosquitto_destroy(mosq);
	mosquitto_lib_cleanup();
	return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mosquitto.h>
#include <ctype.h>


int is_on_db(struct mosquitto *moqst,char * string,char * topic); 


void on_connect(struct mosquitto *mosq, void *obj, int rc) {
  // When connected to the broker
	printf("ID: %d\n", * (int *) obj);
	if(rc) {
		printf("Error with result code: %d\n", rc);
		exit(-1);
	}
	mosquitto_subscribe(mosq, NULL, "request/34/+", 0); // Subscribe to all topics starting with "request/" whatever the ID
  printf("Sub to channel");
}

void on_message(struct mosquitto *mosq, void *obj, const struct mosquitto_message *msg) {
  // When a message is received
	printf("New message with topic %s: %s\n", msg->topic, (char *) msg->payload);
  is_on_db(mosq,msg->payload,msg->topic);  
}

char * get_response_topic(char * topic){
  // Replace "request/ID" by "response/ID"
  char * response_topic=malloc(sizeof(char)*strlen(topic)+sizeof(char)*15);
  response_topic[0]='r';  response_topic[1]='e';  response_topic[2]='s';  response_topic[3]='p';  response_topic[4]='o';  response_topic[5]='n';  response_topic[6]='s'; response_topic[7]='e'; response_topic[8]='/';
  int i=8;
  printf("i=%d\n",i);
  int j=9;
  while(topic[i]!='\0'){
    response_topic[j]=topic[i];
    i++;
    j++;
  }
  response_topic[j]='\0';
  return response_topic;
}


int is_on_db(struct mosquitto *mosq,char * string,char * topic){
  // Return 1 if in db else 0 => Change by calling the func who check in the db

  char * db[]={"localhost/","www.facebook.com","youtube.com","upload.wikimedia.org"}; // Test list => REPLACE BY DB
  char *response_topic=get_response_topic(topic);


  for(int i=0;i<=3;i++){
    if(strstr(db[i],string)){
      mosquitto_publish(mosq, NULL, response_topic,1,"1",0,false); // Publish 1 if found
      printf("Founded !\nPub in %s\n--------------\n",response_topic);
      free(response_topic);
      return 1;  

    }
  }
  mosquitto_publish(mosq, NULL, response_topic,1,"0",0,false); // Publish 0 if not found
  printf("Not found\n Pub in %s\n-------------\n",response_topic);
  free(response_topic);
  return 0;
}

int main() {
	int rc, id=1;

	mosquitto_lib_init();

	struct mosquitto *mosq;

	mosq = mosquitto_new("MQTT-Phishing-Server", true, &id);
	mosquitto_connect_callback_set(mosq, on_connect); // When connected to the broker
	mosquitto_message_callback_set(mosq, on_message); // All messages go trough on_message
	
	rc = mosquitto_connect(mosq, "broker.hivemq.com", 1883, 10); 
	if(rc) {
		printf("Could not connect to Broker with return code %d\n", rc);
		return -1;
	}

	mosquitto_loop_start(mosq); //Listen to messages
	printf("Press Enter to quit...\n");
	getchar();
	mosquitto_loop_stop(mosq, true);

	mosquitto_disconnect(mosq);
	mosquitto_destroy(mosq);
	mosquitto_lib_cleanup();

	return 0;
}

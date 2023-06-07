console.log("Ajout de l'extension dans le navigateur");


var connected_flag = 0;	
var mqtt;
var reconnectTimeout = 2000;
var host = 'broker.hivemq.com'; // hivemq broker for test over the internet
var port = 8000;
var sub_topic = 'response/15'; // Client number (Random one should be used in prod)


var let_go=-1;
function onConnectionLost() 
{
  console.log("Connection lost");
  connected_flag = 0;
}

function onFailure(message) 
{
  console.log("Connection failed");
	setTimeout(MQTTconnect, reconnectTimeout);
}

function onMessageArrived(r_message)
{
	var topic = r_message.destinationName;
	var msg = r_message.payloadString;
	console.log("Message received: " + msg, topic);
  if(msg=="1"){
    console.log("Found in DB => Can't go");
    let_go=1;
  }else{
    console.log("Not found in DB => Can go");
    let_go=0;
  }

}
			
function onConnect()
{
	connected_flag = 1;
  console.log("Connection established");
	mqtt.subscribe(sub_topic);
  console.log("Subscribed to topic: " + sub_topic);
}

function MQTTconnect() 
{
	var x = Math.floor(Math.random() * 10000);  
	var cname = 'web' + x;
	mqtt = new Paho.MQTT.Client(host, port, cname);
	var options ={timeout: 3, onSuccess: onConnect, onFailure: onFailure};
	mqtt.onConnectionLost = onConnectionLost;
	mqtt.onMessageArrived = onMessageArrived
	mqtt.connect(options);
	return false;
}


				
function sendMessage(topic, msg)
{
	if (connected_flag == 0)
	{
		return false;
	}
	var value = msg;
	message = new Paho.MQTT.Message(value);
	message.destinationName = topic;
	mqtt.send(message);
	return false;
}


MQTTconnect();



function cancel(requestDetails) {
  console.log("Intercept : " + requestDetails.url);
  let domain = (new URL(requestDetails.url));

  domain = domain.hostname.replace('www.','');
  console.log(domain); 


  sendMessage('request/15', domain);
  console.log("Sent : " + domain+" in request/15");
  
  while(let_go!==-1){
    if(let_go==1){
      console.log("Cancel : " + requestDetails.url);
      let_go=-1;
      return {cancel: true}; //Block the request
    }if (let_go==0){
      let_go=-1;
      return {cancel: false}; // Do not block the request
    }
  
  }


} 




browser.webRequest.onBeforeRequest.addListener(
  cancel,
  {urls: ["*://*/*"]}, // Match URL with this pattern and go to "cancel" with the matched url
  ["blocking"]
);

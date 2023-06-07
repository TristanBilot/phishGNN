
console.log("Background script running");

//Create an array
var cache = [];

function makeRequest(domain){ //Send request to the HTTP server and return the response (0 or 1)
  const request = new XMLHttpRequest();
  request.open('GET', 'http://localhost:8080/index.html?d='+domain, false);
  request.send(null);

  if (request.status === 200) {
    console.log(request.responseText);
    return request.responseText
  }

}

function cancel(requestDetails) { // Return  {cancel:true} if request should be blocked, else {cancel:false} 

  console.log("Intercept : " + requestDetails.url);
  let domain = (new URL(requestDetails.url));

  domain = domain.hostname.replace('www.','');
  console.log("Domain :" + domain); 
  if(domain=="localhost"){
    //If the matched domain is localhost => We matched our http server request so ignore it (should replace this by the phishing db domain name
    console.log('localhost mathed: ignoring');
    return {cancel:false}; // Do not block this request
  }

  var loadLocalStorage = localStorage.getItem("WhiteList"); // Load whitelisted website (When the user clik on "see it anyway"
  if(loadLocalStorage==null){
    loadLocalStorage="";
  }

  var ArrayList = loadLocalStorage.split(",");
  if(ArrayList.indexOf(domain) > -1){

    console.log("Whitelisted : "+domain);
    return {cancel: false}; // Do not block the request if the domain name is whitelisted
  }

  if (cache.indexOf(domain) > -1) {
    console.log("Already in cache");
    return {cancel: false}; // If domain name is in cache : Do not block it
  }

  //makeRequest(domain);
  console.log("MakeRequest to "+domain);
  answer=makeRequest(domain); // Send request to the HTTP server
  if(answer==0){
    console.log('Good for '+domain);
    cache.push(domain); // Adding domain name to the extension cache if it's not a phishing website
    //console.log(cache)
    return {cancel:false};
  }
  browser.tabs.create({
    url: "stop.html?site="+domain+"&url="+requestDetails.url,
  }); // If it's a phishing website : open a new page showing up an alert message
  console.log("Not good for "+domain);
  return{cancel:true}; // block the request
  
} 


browser.webRequest.onBeforeRequest.addListener(
  cancel,
  {urls: ["*://*/*"]},
  ["blocking"] // Match all urls and go trough "cancel"
);


#!/usr/bin/env python3

from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
import fcntl
import termios
from sys import argv
from subprocess import check_output

def get_pid():
    """
    Return PID of process named 'dico' (phishing db)

    """
    return check_output(["pidof","dico"])

def isonlist(domain):
    """
    Input: domain name
    Output : -1 if process "dico" is not found, 0 if site if not in db, 1 if site is known as phishing site, -2 if error during process

    """

    pid=get_pid().decode('utf-8').strip()
    if pid is None:
        return -1

    with open(f'/proc/{pid}/fd/0', 'w') as fd: #Enter the domain name as an input for "dico" process.
        for char in domain+"\n":
            fcntl.ioctl(fd, termios.TIOCSTI, char)

    f=open("../phising_website_grapper/history.txt","r").read().splitlines()#Read answer from dico
    if domain+"=1" in f:
        return 1
    elif domain+"=0" in f:
        return 0
    else:
        return -2

class S(BaseHTTPRequestHandler):
    """
    Very simple http server

    """

    def _set_response(self):
        """
        Set params for http response
        """
        self.send_response(200)
        self.send_header('Content-type', 'text/html')

        self.end_headers()

    def do_GET(self):
        """
        Handle GET request.


        """
        logging.info("GET request,\nPath: %s\nHeaders:\n%s\n", str(self.path), str(self.headers))
        self._set_response()
        if 'd=' in self.path:
            splitted=self.path.split('=')[1]
            print("DOMAIN",splitted)
            print("RESULT IS ",isonlist(splitted))
        
            logging.info("GET request,\nPath: %s\nHeaders:\n%s\n",
                str(self.path), str(self.headers))

            self.wfile.write("{}".format(str(isonlist(splitted))).encode('utf-8'))
        


    def do_POST(self):
        """
        Handle POST request
        """
        content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
        post_data = self.rfile.read(content_length) # <--- Gets the data itself
        logging.info("POST request,\nPath: %s\nHeaders:\n%s\n\nBody:\n%s\n",
                str(self.path), str(self.headers), post_data.decode('utf-8'))

        self._set_response()
        self.wfile.write("POST request for {}".format(self.path).encode('utf-8'))

def run(server_class=HTTPServer, handler_class=S, port=8080):
    """
    Running http server
    """
    logging.basicConfig(level=logging.INFO)
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    logging.info('Starting httpd...\n')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    logging.info('Stopping httpd...\n')

if __name__ == '__main__':

    if(len(argv)==2):
        run(port=int(argv[1]))
    else:
        run()

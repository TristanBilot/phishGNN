from bs4 import BeautifulSoup
import urllib.request

# GET LINE AND PHISING URL FROM THE WEBSITE


# Send request to get content of the website
url = "https://openphish.com/feed.txt"
ourUrl = urllib.request.urlopen(url)

soup=BeautifulSoup(ourUrl,'html.parser')

text = soup.prettify()
ListeSite = []


i = 0
y = 0

# LIRE DE FICHIER
FileRead = open("phising_site.txt","r")
contenu = FileRead.read()


# Ajout de lignes deja existante dans le fichier phising_site.txt
File = []
for FileLine in contenu.split('\n'):
    File.append(FileLine)


# récupere chaque ligne du site internet
for line in text.split('\n'):
    # si la longueur est de 0 on passe
    if len(line) == 0:
        pass
    else:
        # on supprime la partie "http://"
        Aline = line.split("://", 1)
        Bline = Aline[1]
        Cline = Bline.split("/", 1)
        line = Cline[0]
        # si dans l'url il y a un www. comme dans https://www.google.com on sup^prime cette partie
        if "www." in line :
            Atext = line.split("www.", 1)
            line = Atext[1]
        # Si cette ligne n'est pas dans notre liste des sites de phising qu'on possede deja, on ajoute dans une nouvelle liste
        if line not in File:
            print("[+] Line not in txt file, we can add it ! The line is : ",line)
            ListeSite.append(line)
        else:
            y = y + 1
print("[-] ",y," Line already in txt file !")
        
FileRead.close()
# On tri la liste qu'on a obtenu
ListeSite = sorted(ListeSite)

FileWrite = open("phising_site.txt","a")

# Maintenant on ajoute chaque ligne dans la liste ListeSite dans le fichier txt
for final in ListeSite :
    final += '\n'
    FileWrite.write(final)
    i = i+1
print("[+] ",i,"Line added in the txt file !")




FileWrite.close()

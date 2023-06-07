#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h>

#define _XOPEN_SOURCE 500 
#define TAILLE_MAX 1000

/*
La structure entry_s représente une entrée dans une table de hachage. Elle contient trois champs :

key : un pointeur vers une chaîne de caractères représentant la clé de l'entrée
value : un pointeur vers une chaîne de caractères représentant la valeur associée à la clé
next : un pointeur vers une autre entrée de la table de hachage qui a été placée dans le même compartiment que cette entrée, utilisé pour la gestion des collisions
*/
struct entry_s {
	char *key;
	char *value;
	struct entry_s *next;
};
typedef struct entry_s entry_t;


/*
La structure hashtable_s est une structure pour une table de hachage. Elle contient deux champs :

size : un entier représentant la taille de la table de hachage
table : un pointeur vers un tableau de pointeurs d'entrées, chacun pointant vers le premier élément d'une liste chaînée d'entrées stockées dans le même compartiment
*/
struct hashtable_s {
	int size;
	struct entry_s **table;	
};
typedef struct hashtable_s hashtable_t;



/*
La fonction ht_create est une fonction qui crée une nouvelle table de hachage et l'initialise avec la taille spécifiée
*/
hashtable_t *ht_create( int size ) {
	hashtable_t *hashtable = NULL;
	int i;
	/*Si une allocation de mémoire a echoué alors on renvoie NULL*/
	if( size < 1 ) {
		return NULL;
	}
	if( ( hashtable = malloc( sizeof( hashtable_t ) ) ) == NULL ) {
		return NULL;
	}
	if( ( hashtable->table = malloc( sizeof( entry_t * ) * size ) ) == NULL ) {
		return NULL;
	}

	/*On initialise chaque pointeur du tableau a NULL*/
	for( i = 0; i < size; i++ ) {
		hashtable->table[i] = NULL;
	}
	/*affecte la taille spécifiée à la taille de la table de hachage*/
	hashtable->size = size;
	/*retoure un pointeur vers la nouvelle table de hashage*/
	return hashtable;	
}




/*
Fonction de hachage utilisée pour calculer la position d'une clé dans une table de hachage.
Le premier arguement est la table de hashage
Le deuxième est la clé à calculer son hash

L'algorithme utilise une combinaison de décalages binaires et d'opérations logiques sur les octets de la clé pour calculer un nombre entier unique
*/
int ht_hash( hashtable_t *hashtable, char *key ) {

	unsigned long int hashval;
	int i = 0;

	/*
	La boucle while parcours chaque octet de la clé spécifiée
	À chaque tour de boucle, l'algorithme calcule un nouveau nombre de hachage en décalant le nombre de hachage actuel de 8 bits vers la gauche et en ajoutant l'octet de la clé actuel à hashval
	*/
	while( hashval < ULONG_MAX && i < strlen( key ) ) {
		hashval = hashval << 8;
		hashval += key[ i ];
		i++;
	}
	/*Si le résultat est plus grand que la taille du tableau on fait un modulo de la taille du tableau et on retroune la valeur*/
	return hashval % hashtable->size;
}



/*
La fonction ht_newpair est une fonction qui crée une nouvelle entrée (ou paire clé-valeur) dans une table de hachage
*/
entry_t *ht_newpair( char *key, char *value ) {
	entry_t *newpair;
	/*Alloue de la mémoire pour les nouvelle entrée et si il y a un problème, on renvoie NULL*/
	if( ( newpair = malloc( sizeof( entry_t ) ) ) == NULL ) {
		return NULL;
	}
	if( ( newpair->key = strdup( key ) ) == NULL ) {
		return NULL;
	}
	if( ( newpair->value = strdup( value ) ) == NULL ) {
		return NULL;
	}
	/*initialise le champ next de la nouvelle entrée à NULL, car cette entrée n'a pas encore de collision*/
	newpair->next = NULL;
	/* la fonction renvoie un pointeur vers la nouvelle entrée créée*/
	return newpair;
}





/*
La fonction ht_set ajoute ou met à jour une paire clé-valeur dans une table de hachage
Prend en argument la table de hashage puis une paire de clé, valeur
*/
void ht_set( hashtable_t *hashtable, char *key, char *value ) {
	int bin = 0;
	entry_t *newpair = NULL;
	entry_t *next = NULL;
	entry_t *last = NULL;
	/*On appelle la fonction ht_hash pour determiner à quelle indice du tableau stocker la paire*/
	bin = ht_hash( hashtable, key );

	next = hashtable->table[ bin ];
	/*fonction utilise une boucle while pour parcourir la liste chaînée à l'indice spécifié à la recherche de la paire clé-valeur spécifiée
	Si la paire est trouvée, la fonction met simplement à jour la valeur associée à la clé*/
	while( next != NULL && next->key != NULL && strcmp( key, next->key ) > 0 ) {
		last = next;
		next = next->next;
	}
	if( next != NULL && next->key != NULL && strcmp( key, next->key ) == 0 ) {
		free( next->value );
		next->value = strdup( value );
	} 
	/*Si la paire clé-valeur n'est pas trouvée dans la liste chaînée, la fonction crée une nouvelle paire en appelant la fonction ht_newpair et insère la nouvelle paire dans la liste chaînée
	Ensuite si l'indice du tablear est vide, la nouvelle paire est insérée directement
	Sinon, la nouvelle paire est insérée à la tête de la liste chaînée si la clé est inférieure à toutes les clés existantes dans la liste
	*/
	else {
		newpair = ht_newpair( key, value );
		if( next == hashtable->table[ bin ] ) {
			newpair->next = next;
			hashtable->table[ bin ] = newpair;
		} else if ( next == NULL ) {
			last->next = newpair;
		} else  {
			newpair->next = next;
			last->next = newpair;
		}
	}
}




/*
La fonction ht_get récupère la valeur associée à une clé spécifiée dans une table de hachage
La fonction prend deux argument, le premier la table de hashage et le second la clé à récuperer
*/
char *ht_get( hashtable_t *hashtable, char *key ) {
	int bin = 0;
	entry_t *pair;
	/*On appelle la fonction ht_hash pour determiner à quelle indice du tableau recherche la paire*/
	bin = ht_hash( hashtable, key );

	pair = hashtable->table[ bin ];
	/*la boucle while pour parcourir la liste chaînée à la recherche de la paire clé-valeur rechercher
	si la paire est trouvée, la fonction renvoie la valeur associée à la clé*/
	while( pair != NULL && pair->key != NULL && strcmp( key, pair->key ) > 0 ) {
		pair = pair->next;
	}
	/*Si la paire n'est pas trouvé dans la liste chainée alors la fonction renvoie NULL*/
	if( pair == NULL || pair->key == NULL || strcmp( key, pair->key ) != 0 ) {
		return NULL;
	} else {
		return pair->value;
	}
}


/*
L fonction print_table qui affiche le contenu d'une table de hachage
*/
void print_table(hashtable_t *hashtable)
{
    printf("\nHash Table\n-------------------\n");
	/*La boucle parcours chaque indice du tableau est affiche la clé et la valeur*/
    for (int i = 0; i < hashtable->size; i++)
    {
        if (hashtable -> table[i])
        {
            printf("Index:%d, Key:%s, Value:%s\n", i, hashtable -> table[i] -> key, hashtable -> table[i] -> value);
        }
    }
    printf("-------------------\n\n");
}


/*
La fonction writeFile permet d'écrire dans un fichier la sortie des résultats
*/
int writeFile(int IsPhishing,char * txt){
	FILE *fptr;
   fptr = fopen("history.txt","a");

   if(fptr == NULL)
   {
      printf("Error!");   
      exit(1);             
   }

   if (IsPhishing){
   	fprintf(fptr,"%s=0\n",txt);
   }
   else{
   	fprintf(fptr,"%s=1\n",txt);
   }
   //fprintf(fptr,"%s",txt);
   fclose(fptr);
   return 0;
}



int main() {

  printf("[/!\\] Enter a domain name and check if it is already listed as a phising site.\n");
  printf("[/!\\] If you want to quit this programm type 'quit' in the command line.\n");


  // Create an integer variable that will store the number we get from the user
  char myDomain[50];
  int Bool = 1;

  //Open file
  FILE *f;
  f = fopen("phising_site.txt","r");
  if (NULL == f){
    printf("[-] File can't be opened.\n");
    fclose(f);
  }
  printf("[+] File is loaded !\n");
  fclose(f);

  //Ouverture du fichier txt
  FILE* fichier = NULL;
  char chaine[TAILLE_MAX] = "";

  //initilisation de la hash table
  hashtable_t *hashtable = ht_create( 665536 );
    
  fichier = fopen("phising_site.txt", "rb"); //ouverture du .txt en binaire pour lire les caractères spéciaux

  if(fichier != NULL){
        while(fgets(chaine, TAILLE_MAX, fichier) != NULL){ //lis le fichier ligne par ligne
            if (strcmp(chaine, "\n") == 0){ // Si la chaine contient seulement un saut de ligne on affiche un message
                printf("[-] A Enter space detected \n");
            }
            else{ // Sinon on ajoute la donner dans la table de hashage
                chaine[strlen(chaine)-1]='\0';  
                ht_set( hashtable, chaine, chaine );
            }
        }
        fclose(fichier);
    }
    else{
        printf("[-] Unable to open the file\n");
    }
	/*
	Boucle while s'éxécute tant que le mot clé "quit" n'est pas entrer par l'utilisateur
	*/
  while (Bool == 1)
    {
		/*Demande un nom de domaine a l'utilisateur*/
      printf("[/!\\] Enter a domain name : \n"); 
      scanf("%s", myDomain);
	  /*Si il rentre le mot quit alors le programme s'arrete*/
      if (strcmp(myDomain,"quit") == 0){
        printf("[/!\\] You leave the programm !");
        Bool = 0;
      }
      else{
        /*Si le nom de domain demander par l'utilisateur est un site présent dans la table de hashage alors on informe l'utilisateur*/  
        if (ht_get( hashtable, myDomain ) != NULL){
          printf("[+] The site : %s is a phising site\n", myDomain);
		  writeFile(0, myDomain);
        }
		/*Sinon on renvoie un message*/
        else{
          printf("[-] The site : %s isn't a phising site\n", myDomain);
		  writeFile(1, myDomain);
        }
      }
    }
    return 0;
  }

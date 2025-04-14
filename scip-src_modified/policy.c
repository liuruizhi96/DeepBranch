/*
 * @Author: Ruizhi Liu && liuruizhi19s@ict.ac.cn
 * @Date: 2025-03-26 22:47:26
 * @LastEditors: Ruizhi Liu && liuruizhi19s@ict.ac.cn
 * @LastEditTime: 2025-03-28 17:50:24
 * @FilePath: \scip6_ins_src\scip-6.0.1\src\scip\policy.c
 * @Description: 
 * 
 * Copyright (c) 2025 by ${git_name_email}, All Rights Reserved. 
 */
#define SCIP_DEBUG

/**@file   policy.c
 * @brief  methods for policy
 * @author He He 
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/
#include "scip/def.h"
#include "scip/feat.h"
#include "scip/struct_feat.h"
#include "scip/policy.h"
/// For HashTable
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define BUCKETCOUNT 16
/// HashTable End

#define HEADERSIZE_LIBSVM       6 



SCIP_RETCODE SCIPpolicyCreate(
   SCIP*              scip,
   SCIP_POLICY**      policy
   )
{
   assert(scip != NULL);
   assert(policy != NULL);

   SCIP_CALL( SCIPallocBlockMemory(scip, policy) );
   (*policy)->weights = NULL;
   (*policy)->size = 0;

   return SCIP_OKAY;
}

SCIP_RETCODE SCIPpolicyFree(
   SCIP*              scip,
   SCIP_POLICY**      policy
   )
{
   assert(scip != NULL);
   assert(policy != NULL);

   if ((*policy)->weights != NULL)
   {
      BMSfreeMemoryArray(&(*policy)->weights);
   }
   
   SCIPfreeBlockMemory(scip, policy);

   return SCIP_OKAY;
}

SCIP_RETCODE SCIPreadLIBSVMPolicy(
   SCIP*              scip,
   char*              fname,
   SCIP_POLICY**      policy       
   )
{
   int nlines = 0;
   int i;
   char buffer[SCIP_MAXSTRLEN];
   FILE* file = fopen(fname, "r");
   if( file == NULL )
   {
      SCIPerrorMessage("cannot open file <%s> for reading\n", fname);
      SCIPprintSysError(fname);
      return SCIP_NOFILE;
   }

   /* find out weight vector size */
   while( fgets(buffer, (int)sizeof(buffer), file) != NULL )
      nlines++;
   /* don't count libsvm model header */
   assert(nlines >= HEADERSIZE_LIBSVM);
   (*policy)->size = nlines - HEADERSIZE_LIBSVM;
   fclose(file);
   if( (*policy)->size == 0 )
   {
      SCIPerrorMessage("empty policy model\n");
      return SCIP_NOFILE;
   }

   SCIP_CALL( SCIPallocMemoryArray(scip, &(*policy)->weights, (*policy)->size) );

   /* have to reopen to read weights */
   file = fopen(fname, "r");
   /* skip header */
   for( i = 0; i < HEADERSIZE_LIBSVM; i++ )
      fgets(buffer, (int)sizeof(buffer), file);
   for( i = 0; i < (*policy)->size; i++ )
      fscanf(file, "%"SCIP_REAL_FORMAT, &((*policy)->weights[i]));

   fclose(file);

   SCIPverbMessage(scip, SCIP_VERBLEVEL_NORMAL, NULL, "policy of size %d from file <%s> was %s\n",
      (*policy)->size, fname, "read, will be used in the dagger node selector");

   return SCIP_OKAY;
}

/** calculate score of a node given its feature and the policy weight vector */
void SCIPcalcNodeScore(
   SCIP_NODE*         node,
   SCIP_FEAT*         feat,
   SCIP_POLICY*       policy
   )
{
   int offset = SCIPfeatGetOffset(feat);
   int i;
   SCIP_Real score = 0;
   SCIP_Real* weights = policy->weights;
   SCIP_Real* featvals = SCIPfeatGetVals(feat);

   if( (offset + SCIPfeatGetSize(feat)) > policy->size )
      score = 0;
   else
   {
      for( i = 0; i < SCIPfeatGetSize(feat); i++ )
         score += featvals[i] * weights[i+offset];
   }

   SCIPnodeSetScore(node, score);
   SCIPdebugMessage("score of node  #%"SCIP_LONGINT_FORMAT": %f\n", SCIPnodeGetNumber(node), SCIPnodeGetScore(node));
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/msg.h>
#include "type_definitions.h"

char* substring(char* ch,int pos,int length)  
{  

   char* pch=ch;

   char* subch=(char*)calloc(sizeof(char),length+1);  
   int i;  
 
   pch=pch+pos;  

   for(i=0;i<length;i++)  
   {  
      subch[i]=*(pch++);  
 
   }  
   subch[length]='\0';  
   return subch;        
}

void getStrStartLen(char* str, int* _start, int* _len)
{
   int start = 0;
   int len = 0;
   int i = 2, start_flag = 1;
   
   while (*(str+i) != '\0')
   {
      if (start_flag == 1 && str[i] == '.')
      {
         start = i + 1;
         start_flag = 0;
      }
      else if (start_flag == 0 && str[i] == '.')
      {
         break;
      }
      else if (start_flag == 0)
      {

         assert(str[i] >= '0' && str[i] <= '9');
         len++;
      }
      i++;
   }
   *_start = start;
   *_len = len;
}

SCIP_RETCODE SCIPreadNNPolicy(
   SCIP*             scip,
   char*             fname,
   SCIP_POLICY**     policy
)
{
   //////////////////////////////////////////////////////
   fname = "/home/xuliming/daggerSpace/training_files/scip-dagger/trained_models/setcover/5000_train_1000r_1000c_0.05d/1103liu_scip6_fsb_oracle_node4_06py/2022-11-04-23-38_nodelist/insL200_trjL5e6/searchPolicy.3.bin";
   //////////////////////////////////////////////////////
   char* substr = "searchPolicy";
   char* str = strstr(fname, substr);
   int i = 0, numPolicy = 0;
   while (*(str+i) != '\0')
   {
      if (str[i] >= '0' && str[i] <= '9')
         numPolicy = numPolicy * 10 + str[i] - '0';
      i++;
   }

   (*policy)->numPolicy = numPolicy;
   assert((*policy)->numPolicy >= 0 && (*policy)->numPolicy <= 100000);

   SCIPdebugMessage("numPolicy  #%s %i\n", str, (*policy)->numPolicy);
   return SCIP_OKAY;
}


typedef struct model_server {
	int send_id;
	int receive_id;
} *model_server_t;

model_server_t create_server() {

	model_server_t p_s = (model_server_t) malloc(sizeof(struct model_server));

	if ( -1 == (p_s->receive_id = msgget((key_t)4321, IPC_CREAT | 0666)))
	{
		perror( "msgget() failed");
		exit(1);
	}

	if ( -1 == (p_s->send_id = msgget((key_t)1234, IPC_CREAT | 0666)))
	{
		perror( "msgget() failed");
		exit(1);
	}

	return p_s;
}

void delete_server(model_server_t p_s) {
	free(p_s);
}

void _send_feat(double *data, unsigned data_len, int send_server_id) {
   
	void *databuf = malloc(data_len * sizeof(double) + sizeof(long));
   
	((long *)databuf) [0] = TYPE_ARRAY;

   memcpy(databuf+sizeof(long), data, data_len * sizeof(double));

   if ( -1 == msgsnd(send_server_id, databuf, data_len * sizeof(double), 0))
	{
		perror("msgsnd() failed");
		exit(1);
	}
   
	free(databuf);
}

void _receive_feat(double *data, unsigned data_len, int receive_server_id) {

	void *databuf = malloc(data_len * sizeof(double) + sizeof(long));
	((long *)databuf) [0] = TYPE_ARRAY;

	if ( -1 == msgrcv(receive_server_id, databuf, sizeof(double)*data_len, 0, 0))
	{
		perror("msgrcv() failed");
		exit(1);
	}

	memcpy(data, databuf+sizeof(long), data_len * sizeof(double));
	free(databuf);
}

void call_model(double *data, unsigned data_len, double *out, unsigned out_len, model_server_t p_s) {
   
   _send_feat(data, data_len, p_s->send_id);
	_receive_feat(out, out_len, p_s->receive_id);
}


// huangxulin 2023-08-28
void SCIPcalcNNNodeScore(
   SCIP_NODE*         node1,
   SCIP_FEAT*         feat1,
   SCIP_NODE*         node2,
   SCIP_FEAT*         feat2,
   int               flag
   )
{

   int FEATURE_SIZE = SCIPfeatGetSize(feat1);
   int length = 2 * FEATURE_SIZE;
   double input[length];
   double output[3] = {DBL_MAX};
   model_server_t server = create_server();
   SCIP_Real* left_featvals = SCIPfeatGetVals(feat1);
   SCIP_Real* right_featvals = SCIPfeatGetVals(feat2);

   for (int i = 0; i < FEATURE_SIZE; i ++)
   {
      input[i] = left_featvals[i];
   }

   for (int i = 0; i < FEATURE_SIZE; i++)
   {
      input[i + 20] = right_featvals[i];
   }
   
   while (1)
   {
		call_model(input, length, output, 3, server);
      
      if (output[0] != DBL_MAX)
      {
         if (flag==1)
         {
            SCIPnodeSetScore(node1, output[1]);
            SCIPnodeSetScore(node2, output[2]);
            return ;
         }
         else
         {
            SCIPnodeSetScore(node1, output[1]);
            return ;
         }
         
      }
   }
}


// NN functions
void SCIPcalcNNNodeScoreConcat(
   SCIP_NODE*         node,
   SCIP_FEAT*         feat,
   SCIP_FEAT*         left_feat,
   SCIP_FEAT*         right_feat,
   SCIP_POLICY*       policy
   )
{
   int FEATURE_SIZE = SCIPfeatGetSize(feat);
   int length = 3 * FEATURE_SIZE;
   double input[length + 1];
   double output[2] = {DBL_MAX};
   model_server_t server = create_server();
   SCIP_Real* featvals = SCIPfeatGetVals(feat);
   SCIP_Real* left_featvals = SCIPfeatGetVals(left_feat);
   SCIP_Real* right_featvals = SCIPfeatGetVals(right_feat);

   for (int i = 0; i < FEATURE_SIZE; i ++)
   {
      input[i] = left_featvals[i];
   }

   for (int i = 0; i < FEATURE_SIZE; i++)
   {
      input[i + 20] = right_featvals[i];
   }

   for (int i = 0; i < FEATURE_SIZE; i ++)
   {
      input[i+40] = featvals[i];
   }
   
   input[length] = (double) (policy->numPolicy);

   while (1)
   {
		call_model(input, length + 1, output, 2, server);
      
      if (output[0] != DBL_MAX)
      {
         SCIPnodeSetScore(node, output[1]);
         return ;
      }
   }
}

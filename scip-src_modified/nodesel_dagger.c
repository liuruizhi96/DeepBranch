#define SCIP_DEBUG
/**@file   nodesel_dagger.c
 * @brief  uct node selector which balances exploration and exploitation by considering node visits
 * @author Gregor Hendel
 *
 * the UCT node selection rule selects the next leaf according to a mixed score of the node's actual lower bound
 *
 * The idea of UCT node selection for MIP appeared in:
 *
 * The authors adapted a game-tree exploration scheme called UCB to MIP trees. Starting from the root node as current node,
 *
 * The node selector features several parameters:
 *
 * @note It should be avoided to switch to uct node selection after the branch and bound process has begun because
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/
#include <assert.h>
#include <string.h>

#include "scip/nodesel_dagger.h"
#include "scip/nodesel_oracle.h"
#include "scip/feat.h"
#include "scip/policy.h"
#include "scip/struct_policy.h"
#include "scip/sol.h"
#include "scip/tree.h"
#include "scip/struct_set.h"
#include "scip/struct_scip.h"
#include "time.h"

#include "scip/prob.h"
#include "scip/type_sol.h"
#include "scip/struct_sol.h"

#include "scip/scip_sol.h"
#include "scip/scip_timing.h"

#define NODESEL_NAME            "dagger"
#define NODESEL_DESC            "node selector which selects node according to a policy but writes exampels according to the oracle"
#define NODESEL_STDPRIORITY     10
#define NODESEL_MEMSAVEPRIORITY 0

#define DEFAULT_FILENAME        ""

/*
 * Data structures
 */

/** node selector data */
struct SCIP_NodeselData
{
   char*              solfname;           /**< name of the solution file */
   SCIP_SOL*          optsol;             /**< optimal solution */
   char*              polfname;           /**< name of the solution file */
   SCIP_POLICY*       policy;
   char*              trjfname;           /**< name of the trajectory file */
   FILE*              wfile;
   FILE*              trjfile;
   SCIP_FEAT*         feat;
   SCIP_FEAT*         feat1;              /* pointer to store the features of the first child node */
   SCIP_FEAT*         feat2;              /* pointer to store the features of the second child node */
   SCIP_FEAT*         optfeat;
#ifndef NDEBUG
   SCIP_Longint       optnodenumber;      /**< successively assigned number of the node */
#endif
   SCIP_Bool          negate;
   int                nerrors;            /**< number of wrong ranking of a pair of nodes */
   int                ncomps;              /**< total number of comparisons */

   SCIP_FEAT*         left_feat;
   SCIP_FEAT*         right_feat;
};

void SCIPnodeseldaggerPrintStatistics(
   SCIP*                 scip,
   SCIP_NODESEL*         nodesel,
   FILE*                 file
   )
{
   SCIP_NODESELDATA* nodeseldata;

   assert(scip != NULL);
   assert(nodesel != NULL);

   nodeseldata = SCIPnodeselGetData(nodesel);
   assert(nodeseldata != NULL);

   SCIPmessageFPrintInfo(scip->messagehdlr, file,
         "Node selector      :\n");
   SCIPmessageFPrintInfo(scip->messagehdlr, file,
         "  comp error rate  : %d/%d\n", nodeseldata->nerrors, nodeseldata->ncomps);
   SCIPmessageFPrintInfo(scip->messagehdlr, file,
         "  selection time   : %10.2f\n", SCIPnodeselGetTime(nodesel));
}

/** solving process initialization method of node selector (called when branch and bound process is about to begin) */
static
SCIP_DECL_NODESELINIT(nodeselInitDagger)
{
   SCIP_NODESELDATA* nodeseldata;
   assert(scip != NULL);
   assert(nodesel != NULL);

   nodeseldata = SCIPnodeselGetData(nodesel);
   assert(nodeseldata != NULL);

   /* solfname should be set before including nodeseldagger */
   // nodeseldata->solfname = "/home/xuliming/daggerSpace/training_files/scip-dagger/bfs_solution/cauctions/test_200_1000/instance_30.sol";
   // nodeseldata->solfname = "/home/xuliming/daggerSpace/training_files/scip-dagger/bfs_solution/setcover/test_1000r_1000c_0.05d/instance_1.sol";
   // nodeseldata->solfname = "/home/xuliming/daggerSpace/training_files/scip-dagger/bfs_solution/indset/new_transfer_1000_4/instance_102.sol";
   nodeseldata->solfname = "/home/xuliming/daggerSpace/training_files/scip-dagger/bfs_solution/facilities/transfer_400_100_5/instance_87.sol";
   assert(nodeseldata->solfname != NULL);
   nodeseldata->optsol = NULL;
   SCIP_CALL( SCIPreadOptSol(scip, nodeseldata->solfname, &nodeseldata->optsol) );
   assert(nodeseldata->optsol != NULL);
#ifdef SCIP_DEBUG
   SCIP_CALL( SCIPprintSol(scip, nodeseldata->optsol, NULL, FALSE) );
#endif

   /* read policy */
   SCIP_CALL( SCIPpolicyCreate(scip, &nodeseldata->policy) );
   assert(nodeseldata->polfname != NULL);
   // SCIP_CALL( SCIPreadLIBSVMPolicy(scip, nodeseldata->polfname, &nodeseldata->policy) );

   //liu
   SCIP_CALL( SCIPreadNNPolicy(scip, nodeseldata->polfname, &nodeseldata->policy) );
   // assert(nodeseldata->policy->weights != NULL); // xuliming: NN policy has no weights

   /* open trajectory file for writing */
   /* open in appending mode for writing training file from multiple problems */
   nodeseldata->trjfile = NULL;
   if( nodeseldata->trjfname != NULL )
   {
      char wfname[SCIP_MAXSTRLEN];
      strcpy(wfname, nodeseldata->trjfname);
      strcat(wfname, ".weight");
      nodeseldata->wfile = fopen(wfname, "a");
      nodeseldata->trjfile = fopen(nodeseldata->trjfname, "a");
   }

   /* create feat */
   nodeseldata->feat = NULL;
   SCIP_CALL( SCIPfeatCreate(scip, &nodeseldata->feat, SCIP_FEATNODESEL_SIZE) );
   // SCIP_CALL( SCIPhgfeatCreate(scip, &nodeseldata->feat, SCIP_FEATNODESEL_SIZE) );
   assert(nodeseldata->feat != NULL);
   SCIPfeatSetMaxDepth(nodeseldata->feat, SCIPgetNBinVars(scip) + SCIPgetNIntVars(scip));

   /* create feature vector for first child node */
   nodeseldata->feat1 = NULL;
   SCIP_CALL( SCIPfeatCreate(scip, &(*nodeseldata).feat1, SCIP_FEATNODESEL_SIZE) );
   assert(nodeseldata->feat1 != NULL);
   SCIPfeatSetMaxDepth(nodeseldata->feat1, SCIPgetNBinVars(scip) + SCIPgetNIntVars(scip));

   /* create feature vector for second child node */
   nodeseldata->feat2 = NULL;
   SCIP_CALL( SCIPfeatCreate(scip, &(*nodeseldata).feat2, SCIP_FEATNODESEL_SIZE) );
   assert(nodeseldata->feat2 != NULL);
   SCIPfeatSetMaxDepth(nodeseldata->feat2, SCIPgetNBinVars(scip) + SCIPgetNIntVars(scip));
   
   /* create root feat */
   nodeseldata->left_feat = NULL;
   SCIP_CALL( SCIPfeatCreate(scip, &nodeseldata->left_feat, SCIP_FEATNODESEL_SIZE) );
   assert(nodeseldata->left_feat != NULL);
   SCIPfeatSetMaxDepth(nodeseldata->left_feat, SCIPgetNBinVars(scip) + SCIPgetNIntVars(scip));

   nodeseldata->right_feat = NULL;
   SCIP_CALL( SCIPfeatCreate(scip, &nodeseldata->right_feat, SCIP_FEATNODESEL_SIZE) );
   assert(nodeseldata->right_feat != NULL);
   SCIPfeatSetMaxDepth(nodeseldata->right_feat, SCIPgetNBinVars(scip) + SCIPgetNIntVars(scip));

   /* create optimal node feat */
   nodeseldata->optfeat = NULL;
   SCIP_CALL( SCIPfeatCreate(scip, &nodeseldata->optfeat, SCIP_FEATNODESEL_SIZE) );
   // SCIP_CALL( SCIPhgeatCreate(scip, &nodeseldata->optfeat, SCIP_FEATNODESEL_SIZE) );
   assert(nodeseldata->optfeat != NULL);
   SCIPfeatSetMaxDepth(nodeseldata->optfeat, SCIPgetNBinVars(scip) + SCIPgetNIntVars(scip));

#ifndef NDEBUG
   nodeseldata->optnodenumber = -1;
#endif
   nodeseldata->negate = TRUE;

   nodeseldata->nerrors = 0;
   nodeseldata->ncomps = 0;

   return SCIP_OKAY;
}

/** destructor of node selector to free user data (called when SCIP is exiting) */
static
SCIP_DECL_NODESELEXIT(nodeselExitDagger)
{
   SCIP_NODESELDATA* nodeseldata;
   assert(scip != NULL);
   assert(nodesel != NULL);

   nodeseldata = SCIPnodeselGetData(nodesel);

   assert(nodeseldata->optsol != NULL);
   SCIP_CALL( SCIPfreeSolSelf(scip, &nodeseldata->optsol) );

   if( nodeseldata->trjfile != NULL)
   {
      fclose(nodeseldata->wfile);
      fclose(nodeseldata->trjfile);
   }

   assert(nodeseldata->feat != NULL);
   SCIP_CALL( SCIPfeatFree(scip, &nodeseldata->feat) );

   assert(nodeseldata->feat1 != NULL);
   SCIP_CALL( SCIPfeatFree(scip, &nodeseldata->feat1) );

   assert(nodeseldata->feat2 != NULL);
   SCIP_CALL( SCIPfeatFree(scip, &nodeseldata->feat2) );

   assert(nodeseldata->left_feat != NULL);
   SCIP_CALL( SCIPfeatFree(scip, &nodeseldata->left_feat) );
   
   assert(nodeseldata->right_feat != NULL);
   SCIP_CALL( SCIPfeatFree(scip, &nodeseldata->right_feat) );
   
   if( nodeseldata->optfeat != NULL )
      SCIP_CALL( SCIPfeatFree(scip, &nodeseldata->optfeat) );

   assert(nodeseldata->policy != NULL);
   SCIP_CALL( SCIPpolicyFree(scip, &nodeseldata->policy) );

   return SCIP_OKAY;
}

/** destructor of node selector to free user data (called when SCIP is exiting) */
static
SCIP_DECL_NODESELFREE(nodeselFreeDagger)
{
   SCIP_NODESELDATA* nodeseldata;

   nodeseldata = SCIPnodeselGetData(nodesel);
   assert(nodeseldata != NULL);

   SCIPfreeBlockMemory(scip, &nodeseldata);

   SCIPnodeselSetData(nodesel, NULL);

   return SCIP_OKAY;
}

/** 1124 xuliming new SCIP_DECL_NODESELSELECT() using SCIPfeatNNPrint() / node selection method of node selector */
static
SCIP_DECL_NODESELSELECT(nodeselSelectDagger)
{
   SCIP_NODESELDATA* nodeseldata;
   SCIP_NODE** leaves;
   SCIP_NODE** children;
   SCIP_NODE** siblings;
   int nleaves;
   int nsiblings;
   int nchildren;
   int optchild;
   int i;

   int smaller_nodes = 0;
   SCIP_Real opt_score = -100.0;
   SCIP_Real max_score = -100.0;

   assert(nodesel != NULL);
   assert(strcmp(SCIPnodeselGetName(nodesel), NODESEL_NAME) == 0);
   assert(scip != NULL);
   assert(selnode != NULL);

   nodeseldata = SCIPnodeselGetData(nodesel);
   assert(nodeseldata != NULL);

   /* collect leaves, children and siblings data */
   SCIP_CALL( SCIPgetOpenNodesData(scip, &leaves, &children, &siblings, &nleaves, &nchildren, &nsiblings) );

   /* check newly created nodes */
   optchild = -1;
//    for( i = 0; i < nchildren; i++)
//    {
//       /* compute score */
//       SCIPcalcNodeselFeat(scip, children[i], nodeseldata->feat);
//       SCIPcalcNNNodeScore(children[i], nodeseldata->feat, nodeseldata->policy);
//       // SCIPcalcNNNodeScoreConcat(children[i], nodeseldata->feat, nodeseldata->left_feat, nodeseldata->right_feat, nodeseldata->policy);

//       /* check optimality */
//       if( ! SCIPnodeIsOptchecked(children[i]) )
//       {
//          SCIPnodeCheckOptimal(scip, children[i], nodeseldata->optsol);
//          SCIPnodeSetOptchecked(children[i]);
//       }
//       if( SCIPnodeIsOptimal(children[i]) )
//       {
// #ifndef NDEBUG
//          // SCIPdebugMessage("opt node #%"SCIP_LONGINT_FORMAT"\n", SCIPnodeGetNumber(children[i]));
//          nodeseldata->optnodenumber = SCIPnodeGetNumber(children[i]);
// #endif
//          optchild = i;
//       }
//       // if (SCIPnodeGetNumber(children[i]) == 2)
//       //    SCIPcalcNodeselFeat(scip, children[i], nodeseldata->left_feat);
//       // if (SCIPnodeGetNumber(children[i]) == 3)
//       //    SCIPcalcNodeselFeat(scip, children[i], nodeseldata->right_feat);
//    }
   /* check newly created nodes huangxulin 2023-8-28*/
   for (i = 0; i < nchildren; i += 2) // Increment i by 2 to handle two children at once
   {
      /* compute features for the two children */
      SCIPcalcNodeselFeat(scip, children[i], nodeseldata->feat1);
      SCIPcalcNodeselFeat(scip, children[i + 1], nodeseldata->feat2);

      /* compute scores for the two children */
      SCIPcalcNNNodeScore(children[i], nodeseldata->feat1, children[i + 1], nodeseldata->feat2, 1);

      /* check optimality for the first child */
      if (!SCIPnodeIsOptchecked(children[i]))
      {
         SCIPnodeCheckOptimal(scip, children[i], nodeseldata->optsol);
         SCIPnodeSetOptchecked(children[i]);
      }
      if (SCIPnodeIsOptimal(children[i]))
      {
#ifndef NDEBUG
      // SCIPdebugMessage("opt node #%"SCIP_LONGINT_FORMAT"\n", SCIPnodeGetNumber(children[i]));
      nodeseldata->optnodenumber = SCIPnodeGetNumber(children[i]);
#endif
         optchild = i;
      }

      /* check optimality for the second child */
      if (!SCIPnodeIsOptchecked(children[i + 1]))
      {
         SCIPnodeCheckOptimal(scip, children[i + 1], nodeseldata->optsol);
         SCIPnodeSetOptchecked(children[i + 1]);
      }
      if (SCIPnodeIsOptimal(children[i + 1]))
      {
#ifndef NDEBUG
      // SCIPdebugMessage("opt node #%"SCIP_LONGINT_FORMAT"\n", SCIPnodeGetNumber(children[i + 1]));
      nodeseldata->optnodenumber = SCIPnodeGetNumber(children[i + 1]);
#endif
         optchild = i + 1;
      }
   }

   /* check if nchildren is odd */
   if (nchildren % 2 == 1)
   {
      /* compute features for the last child */
      SCIPcalcNodeselFeat(scip, children[nchildren - 1], nodeseldata->feat1);
      SCIPcalcNodeselFeat(scip, children[nchildren - 2], nodeseldata->feat2);
      /* compute score for the last child */
      SCIPcalcNNNodeScore(children[nchildren - 1], nodeseldata->feat1, children[nchildren - 2], nodeseldata->feat2, 0);

      /* check optimality for the last child */
      if (!SCIPnodeIsOptchecked(children[nchildren - 1]))
      {
         SCIPnodeCheckOptimal(scip, children[nchildren - 1], nodeseldata->optsol);
         SCIPnodeSetOptchecked(children[nchildren - 1]);
      }
      if (SCIPnodeIsOptimal(children[nchildren - 1]))
      {
#ifndef NDEBUG
      // SCIPdebugMessage("opt node #%"SCIP_LONGINT_FORMAT"\n", SCIPnodeGetNumber(children[nchildren - 1]));
      nodeseldata->optnodenumber = SCIPnodeGetNumber(children[nchildren - 1]);
#endif
         optchild = nchildren - 1;
      }
   }

   /* write examples */
   if( nodeseldata->trjfile != NULL)
   {
      /* new opt*/
      if( optchild != -1 )
      {
         SCIPcalcNodeselFeat(scip, children[optchild], nodeseldata->optfeat);
         for( i = 0; i < nchildren; i++)
         {
            if( i != optchild )
            {
               SCIPcalcNodeselFeat(scip, children[i], nodeseldata->feat);
#ifndef SCIP_DEBUG
               SCIPdebugMessage("example  #%d #%d\n", (int)nodeseldata->optnodenumber, (int)SCIPnodeGetNumber(children[i]));
#endif
               SCIPfeatNNPrint(scip, nodeseldata->trjfile, nodeseldata->wfile, nodeseldata->feat, -1, nodeseldata->negate);
            }
            else
            {
               SCIPfeatNNPrint(scip, nodeseldata->trjfile, nodeseldata->wfile, nodeseldata->optfeat, 1, nodeseldata->negate);
            }
         }
         for ( i = 0; i < nsiblings; i++)
         {
            SCIPcalcNodeselFeat(scip, siblings[i], nodeseldata->feat);
#ifndef SCIP_DEBUG
            SCIPdebugMessage("example  #%d #%d\n", (int)nodeseldata->optnodenumber, (int)SCIPnodeGetNumber(siblings[i]));
#endif
            SCIPfeatNNPrint(scip, nodeseldata->trjfile, nodeseldata->wfile, nodeseldata->feat, -1, nodeseldata->negate);
         }
         for (i = 0; i < nleaves; i++)
         {
            SCIPcalcNodeselFeat(scip, leaves[i], nodeseldata->feat);
#ifndef SCIP_DEBUG
            SCIPdebugMessage("example  #%d #%d\n", (int)nodeseldata->optnodenumber, (int)SCIPnodeGetNumber(leaves[i]));
#endif
            SCIPfeatNNPrint(scip, nodeseldata->trjfile, nodeseldata->wfile, nodeseldata->feat, -1, nodeseldata->negate);
         }
      }
      else
      {
         /* children are not optimal */
#ifndef SCIP_DEBUG
         assert(nchildren == 0 || (nchildren > 0 && nodeseldata->optnodenumber != -1));
#endif
         for( i = 0; i < nchildren; i++ )
         {
            SCIPcalcNodeselFeat(scip, children[i], nodeseldata->feat);
#ifndef SCIP_DEBUG
            SCIPdebugMessage("example  #%d #%d\n", (int)nodeseldata->optnodenumber, (int)SCIPnodeGetNumber(children[i]));
#endif
            SCIPfeatNNPrint(scip, nodeseldata->trjfile, nodeseldata->wfile, nodeseldata->feat, -1, nodeseldata->negate);
         }
      }
   }

   if (TRUE)
   {
      SCIP_Real opt_score = -100.00;
      SCIP_Real max_score = -100.00;
      int i; int smaller_nodes = 0;
      for( i = 0; i < nchildren; i++)
         if (children[i] != NULL)
            opt_score = SCIPnodeIsOptimal(children[i]) ? SCIPnodeGetScore(children[i]) : opt_score;
      for ( i = 0; i < nsiblings; i++)
         if (siblings[i] != NULL)
            opt_score = SCIPnodeIsOptimal(siblings[i]) ? SCIPnodeGetScore(siblings[i]) : opt_score;
      for (i = 0; i < nleaves; i++)
         if (leaves[i] != NULL)
            opt_score = SCIPnodeIsOptimal(leaves[i]) ? SCIPnodeGetScore(leaves[i]) : opt_score;

      for ( i = 0; i < nchildren; i++){
         if (children[i] != NULL)
         {
            smaller_nodes += SCIPisLE(scip, SCIPnodeGetScore(children[i]), opt_score) ? 1 : 0;
            max_score = SCIPisLE(scip, max_score, SCIPnodeGetScore(children[i])) ? SCIPnodeGetScore(children[i]) : max_score;
         }
      }
      for( i = 0; i < nsiblings; i++){
         if (siblings[i] != NULL)
         {
            smaller_nodes += SCIPisLE(scip, SCIPnodeGetScore(siblings[i]), opt_score) ? 1 : 0;   
            max_score = SCIPisLE(scip, max_score, SCIPnodeGetScore(siblings[i])) ? SCIPnodeGetScore(siblings[i]) : max_score;
         }
      }
      for( i = 0; i < nleaves; i++)
         if (leaves[i] != NULL)
         {
            smaller_nodes += SCIPisLE(scip, SCIPnodeGetScore(leaves[i]), opt_score) ? 1 : 0;
            max_score = SCIPisLE(scip, max_score, SCIPnodeGetScore(leaves[i])) ? SCIPnodeGetScore(leaves[i]) : max_score;
         }
      if (SCIPisEQ(scip, opt_score, -100.0))
         SCIPdebugMessage("opt is not in PQ, total %d max_score %.3f\n", nleaves+nchildren+nsiblings, max_score);
      else
         SCIPdebugMessage("opt is better than %d total %d max_score %.3f opt_score %.3f\n", smaller_nodes, nleaves+nchildren+nsiblings, max_score, opt_score);
   }

   /* experiment start: ml-HeFeatures */
   *selnode = SCIPgetBestNode(scip);

   if (TRUE)
   {
      int idx = -1;
      int isopt = -1;
      int depth = -1;
      SCIP_Real score = -1.0;
      int left = nchildren + nsiblings + nleaves;
      SCIP_SOL *bestSol=NULL;
      SCIP_Real lowerbound = -1;
      SCIP_Real dualbound = SCIPgetDualbound(scip);
      SCIP_Real primalbound = SCIPgetPrimalbound(scip);
      
      // time_t time_ptr;
      // time(&time_ptr);
      SCIP_Real bPB_time = 0.0;
      SCIP_Real current_time = SCIPgetSolvingTime(scip);
      SCIP_Real total_time = SCIPgetTotalTime(scip);
      int nsols = SCIPgetNSols(scip);
      int s;

      int stage = -1;
      // time_t time_ptr;
      // time(&time_ptr);
      bestSol = SCIPgetBestSol(scip);
      if (bestSol != NULL) {
        bPB_time = SCIPgetSolTime(scip, bestSol);
      }

      stage = SCIPgetStage(scip);

      if (*selnode != NULL)
      {
         SCIPprintNodeRootPath(scip, *selnode, NULL);        
         idx = SCIPnodeGetNumber(*selnode);
        
         if (SCIPnodeIsOptimal(*selnode))
            isopt = 1;
        
         depth = SCIPnodeGetDepth(*selnode);
         score = SCIPnodeGetScore(*selnode);
         lowerbound = SCIPnodeGetLowerbound(*selnode);
      }
      SCIPdebugMessage("final selecting node number %d primalbound %.3f lowerbound %.3f dualbound %.3f current_time %.2f depth %d left %d bPB_time %.2f opt %d score %.3f nsols %d ",
                                                   idx, primalbound, lowerbound, dualbound, current_time, depth, left, bPB_time, isopt, score, nsols);

      if ( *selnode != NULL)
      {
         SCIP_SOL** sols_top5;
         sols_top5 = SCIPgetSols(scip);

         for (s = 0; s < 5 && s < nsols; ++s)
         {
            SCIP_SOL* sol;
            SCIP_Real objvalue = -1.0;
            sol = sols_top5[s];

            if( SCIPsolIsOriginal(sol) )
               objvalue = SCIPsolGetOrigObj(sol);
            else
               objvalue = SCIPprobExternObjval(scip->transprob, scip->origprob, scip->set, SCIPsolGetObj(sol, scip->set, scip->transprob, scip->origprob));
            
            SCIPinfoMessage(scip, NULL, "obj%d %.0f ", s, objvalue);
         }
      }
      SCIPinfoMessage(scip, NULL, "total_time %.2f", total_time);
      SCIPinfoMessage(scip, NULL, " \n");
   
      if ( *selnode == NULL)
      {
         SCIPdebugMessage("selnode == NULL \n");
         SCIP_SOL** sols;
         sols = SCIPgetSols(scip);

         for (s = 0; s < nsols && s < 10; ++s)
         {
            SCIP_SOL* sol;
            sol = sols[s];
            SCIPprintSol(scip, sol, NULL, FALSE);
         }
         
         SCIPdebugMessage("SCIPgetSolNodenum %lld, SCIPgetNBestSolsFound %lld\n", SCIPgetSolNodenum(scip, SCIPgetBestSol(scip)), SCIPgetNBestSolsFound(scip));
      }
   }

   if (FALSE)
      SCIPdebugMessage("Selecting node number %lld\n", *selnode != NULL ? SCIPnodeGetNumber(*selnode) : -1l);

   /* experiment end */

   return SCIP_OKAY;
}


/** node comparison method of dagger node selector */
/** compare with 0.5 */
static
SCIP_DECL_NODESELCOMP(nodeselCompDagger)
{  /*lint --e{715}*/
   SCIP_Real score1;
   SCIP_Real score2;
   SCIP_Bool isopt1;
   SCIP_Bool isopt2;
   SCIP_NODESELDATA* nodeseldata;
   int result;
   // SCIP_Real mid = 0.5;

   int node1_idx = SCIPnodeGetNumber(node1);
   int node2_idx = SCIPnodeGetNumber(node2);

   assert(nodesel != NULL);
   assert(strcmp(SCIPnodeselGetName(nodesel), NODESEL_NAME) == 0);
   assert(scip != NULL);

   assert(SCIPnodeIsOptchecked(node1) == TRUE);
   assert(SCIPnodeIsOptchecked(node2) == TRUE);

   score1 = SCIPnodeGetScore(node1);
   score2 = SCIPnodeGetScore(node2);

   assert(score1 != 0);
   assert(score2 != 0);

   /* 根节点的两个子节点永远优于优先队列中其余节点 */
   if (node1_idx <= 3 && node2_idx > 3)
      result = -1;
   else if (node2_idx <= 3 && node1_idx > 3)
      result = 1;

   /* score1 greater than score2 */
   if ( SCIPisGT(scip, score1, score2) )
      result = -1;
   else if (SCIPisLT(scip, score1, score2))
      result = +1;
   // if (SCIPisGT(scip, score1, score2) && SCIPisGT(scip, score1, mid))
   //    result = -1;
   // /* if val1 is (more than epsilon) lower than val2 */
   // else if (SCIPisLT(scip, score1, score2) && SCIPisLT(scip, mid, score2))
   //    result = +1;
   
   

   /* 差值小于eps（set设置的值） close */
   else
   {
      /* experiment xuliming change comp */
      if (FALSE)
      {
         SCIP_Real lowerbound1;
         SCIP_Real lowerbound2;

         assert(nodesel != NULL);
         assert(strcmp(SCIPnodeselGetName(nodesel), NODESEL_NAME) == 0);
         assert(scip != NULL);

         lowerbound1 = SCIPnodeGetLowerbound(node1);
         lowerbound2 = SCIPnodeGetLowerbound(node2);
         if( SCIPisLT(scip, lowerbound1, lowerbound2) )
            result = -1;
         else if( SCIPisGT(scip, lowerbound1, lowerbound2) )
            result = +1;
         else
         {
            SCIP_Real estimate1;
            SCIP_Real estimate2;

            estimate1 = SCIPnodeGetEstimate(node1);
            estimate2 = SCIPnodeGetEstimate(node2);
            if( (SCIPisInfinity(scip,  estimate1) && SCIPisInfinity(scip,  estimate2)) ||
                (SCIPisInfinity(scip, -estimate1) && SCIPisInfinity(scip, -estimate2)) ||
                SCIPisEQ(scip, estimate1, estimate2) )
            {
               SCIP_NODETYPE nodetype1;
               SCIP_NODETYPE nodetype2;

               nodetype1 = SCIPnodeGetType(node1);
               nodetype2 = SCIPnodeGetType(node2);
               if( nodetype1 == SCIP_NODETYPE_CHILD && nodetype2 != SCIP_NODETYPE_CHILD )
                  result = -1;
               else if( nodetype1 != SCIP_NODETYPE_CHILD && nodetype2 == SCIP_NODETYPE_CHILD )
                  result = +1;
               else if( nodetype1 == SCIP_NODETYPE_SIBLING && nodetype2 != SCIP_NODETYPE_SIBLING )
                  result = -1;
               else if( nodetype1 != SCIP_NODETYPE_SIBLING && nodetype2 == SCIP_NODETYPE_SIBLING )
                  result = +1;
               else
               {
                  int depth1;
                  int depth2;

                  depth1 = SCIPnodeGetDepth(node1);
                  depth2 = SCIPnodeGetDepth(node2);
                  if( depth1 < depth2 )
                     result = -1;
                  else if( depth1 > depth2 )
                     result = +1;
                  else
                     result = 0;
               }
            }

            if( SCIPisLT(scip, estimate1, estimate2) )
               result = -1;

            assert(SCIPisGT(scip, estimate1, estimate2));
            result = +1;
         }
      }

      else
      {
         int depth1;
         int depth2;

         depth1 = SCIPnodeGetDepth(node1);
         depth2 = SCIPnodeGetDepth(node2);
         if( depth1 > depth2 )
            result = -1;
         else if( depth1 < depth2 )
            result = +1;
         else
         {
            SCIP_Real lowerbound1;
            SCIP_Real lowerbound2;

            lowerbound1 = SCIPnodeGetLowerbound(node1);
            lowerbound2 = SCIPnodeGetLowerbound(node2);
            if( SCIPisLT(scip, lowerbound1, lowerbound2) )
               result = -1;
            else if( SCIPisGT(scip, lowerbound1, lowerbound2) )
               result = +1;
            else
               result = 0;
         }         
      }
   }

   nodeseldata = SCIPnodeselGetData(nodesel);
   assert(nodeseldata != NULL);
   isopt1 = SCIPnodeIsOptimal(node1);
   isopt2 = SCIPnodeIsOptimal(node2);
   if( (isopt1 && result == 1) || (isopt2 && result == -1) )
      nodeseldata->nerrors++;
   /* don't count the case when neither node is opt, in which case it will always be correct */
   if( isopt1 || isopt2 )
      nodeseldata->ncomps++;

   return result;
}

/*
 * node selector specific interface methods
 */

/** creates the uct node selector and includes it in SCIP */
SCIP_RETCODE SCIPincludeNodeselDagger(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_NODESELDATA* nodeseldata;
   SCIP_NODESEL* nodesel;

   /* create dagger node selector data */
   SCIP_CALL( SCIPallocBlockMemory(scip, &nodeseldata) );

   nodesel = NULL;
   nodeseldata->optsol = NULL;
   nodeseldata->solfname = NULL;
   nodeseldata->trjfname = NULL;
   nodeseldata->polfname = NULL;

   /* use SCIPincludeNodeselBasic() plus setter functions if you want to set callbacks one-by-one and your code should
    * compile independent of new callbacks being added in future SCIP versions
    */
   SCIP_CALL( SCIPincludeNodeselBasic(scip, &nodesel, NODESEL_NAME, NODESEL_DESC, NODESEL_STDPRIORITY,
          NODESEL_MEMSAVEPRIORITY, nodeselSelectDagger, nodeselCompDagger, nodeseldata) );

   assert(nodesel != NULL);

   /* set non fundamental callbacks via setter functions */
   SCIP_CALL( SCIPsetNodeselCopy(scip, nodesel, NULL) );
   SCIP_CALL( SCIPsetNodeselInit(scip, nodesel, nodeselInitDagger) );
   SCIP_CALL( SCIPsetNodeselExit(scip, nodesel, nodeselExitDagger) );
   SCIP_CALL( SCIPsetNodeselFree(scip, nodesel, nodeselFreeDagger) );

   /* add dagger node selector parameters */
   SCIP_CALL( SCIPaddStringParam(scip,
         "nodeselection/"NODESEL_NAME"/solfname",
         "name of the optimal solution file",
         &nodeseldata->solfname, FALSE, DEFAULT_FILENAME, NULL, NULL) );
   SCIP_CALL( SCIPaddStringParam(scip,
         "nodeselection/"NODESEL_NAME"/trjfname",
         "name of the file to write node selection trajectories",
         &nodeseldata->trjfname, FALSE, DEFAULT_FILENAME, NULL, NULL) );
   SCIP_CALL( SCIPaddStringParam(scip,
         "nodeselection/"NODESEL_NAME"/polfname",
         "name of the policy model file",
         &nodeseldata->polfname, FALSE, DEFAULT_FILENAME, NULL, NULL) );

   return SCIP_OKAY;
}

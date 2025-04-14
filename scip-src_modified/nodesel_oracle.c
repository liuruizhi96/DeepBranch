#define SCIP_DEBUG
/**@file   nodesel_uct.c
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
#include "scip/nodesel_oracle.h"
#include "scip/feat.h"
#include "scip/sol.h"
#include "scip/tree.h"
#include "scip/struct_set.h"
#include "scip/struct_scip.h"

#include "scip/type_sol.h"
#include "scip/struct_sol.h"
#include "scip/prob.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "scip/scip_sol.h"
#include "scip/scip_timing.h"

#define NODESEL_NAME            "oracle"
#define NODESEL_DESC            "node selector which always selects the optimal node"
//#define NODESEL_STDPRIORITY     10000000 /* 10000000 */
#define NODESEL_STDPRIORITY     100 /* 10000000 */
#define NODESEL_MEMSAVEPRIORITY 0

#define DEFAULT_FILENAME        ""

/*
 * Data structures
 */

/** node selector data */
struct SCIP_NodeselData
{
   SCIP_SOL*          optsol;             /**< optimal solution */
   char*              solfname;           /**< name of the solution file */
   char*              trjfname;           /**< name of the trajectory file */
   FILE*              trjfile;
   FILE*              wfile;
   SCIP_FEAT*         feat;
   SCIP_FEAT*         optfeat;
#ifndef NDEBUG
   SCIP_Longint       optnodenumber;      /**< successively assigned number of the node */
#endif
   SCIP_Bool          negate;

   SCIP_FEAT*         left_feat;
   SCIP_FEAT*         right_feat;

   SCIP_Real          obj_so_far;       /**< xuliming: obj so far */
   int                cur_group_idx;    /**< xuliming: current group index */
};


/*
 * Local methods
 */

/** check if the given node include the optimal solution */
/* TODO: remove to ischecked */
SCIP_RETCODE SCIPnodeCheckOptimal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NODE*            node,               /**< the node in question */
   SCIP_SOL*             optsol              /**< node selector data */
   )
{
   /* get parent branch vars lead to this node */

   SCIP_VAR**            branchvars;         /* array of variables on which the branchings has been performed in all ancestors */
   SCIP_Real*            branchbounds;       /* array of bounds which the branchings in all ancestors set */
   SCIP_BOUNDTYPE*       boundtypes;         /* array of boundtypes which the branchings in all ancestors set */
   int                   nbranchvars;        /* number of variables on which branchings have been performed in all ancestors
                                              *   if this is larger than the array size, arrays should be reallocated and method should be called again */
   int                   branchvarssize;     /* available slots in arrays */

   int i;
   SCIP_NODE* parent;
   SCIP_Bool isoptimal = TRUE;


   assert(optsol != NULL);
   assert(node != NULL);

   // SCIPdebugMessage("checking node %d\n", (int)SCIPnodeGetNumber(node));

   assert(SCIPnodeIsOptchecked(node) == FALSE);

   /* don't consider root node */
   assert(SCIPnodeGetDepth(node) != 0);
   
   if (TRUE)
   {
      SCIP_Real nodelowerbound = SCIPnodeGetLowerbound(node);
      SCIP_Real primalbound = SCIPgetPrimalbound(scip);
      SCIP_Real dualbound = SCIPgetDualbound(scip);
      int idx = (int)SCIPnodeGetNumber(node);

      SCIPdebugMessage("checking node %d pb %.3f db %.3f nlb %.3f\n", idx, primalbound, dualbound, nodelowerbound);
   }

   /* check parent: if parent is not optimal, its subtree is not optimal */
   parent = SCIPnodeGetParent(node);
   /* root is always optimal */
   if( SCIPnodeGetDepth(parent) > 0 && !SCIPnodeIsOptimal(parent) )
      return SCIP_OKAY;

   branchvarssize = 1;

   /* memory allocation */
   SCIP_CALL( SCIPallocBufferArray(scip, &branchvars, branchvarssize) );
   SCIP_CALL( SCIPallocBufferArray(scip, &branchbounds, branchvarssize) );
   SCIP_CALL( SCIPallocBufferArray(scip, &boundtypes, branchvarssize) );

   SCIPnodeGetParentBranchings(node, branchvars, branchbounds, boundtypes, &nbranchvars, branchvarssize);

   /* if the arrays were to small, we have to reallocate them and recall SCIPnodeGetParentBranchings */
   if( nbranchvars > branchvarssize )
   {
      branchvarssize = nbranchvars;

      /* memory reallocation */
      SCIP_CALL( SCIPreallocBufferArray(scip, &branchvars, branchvarssize) );
      SCIP_CALL( SCIPreallocBufferArray(scip, &branchbounds, branchvarssize) );
      SCIP_CALL( SCIPreallocBufferArray(scip, &boundtypes, branchvarssize) );

      SCIPnodeGetParentBranchings(node, branchvars, branchbounds, boundtypes, &nbranchvars, branchvarssize);
      assert(nbranchvars == branchvarssize);
   }

   /* check optimality */
   assert(nbranchvars >= 1);
   for( i = 0; i < nbranchvars; ++i)
   {
      SCIP_Real optval = SCIPgetSolVal(scip, optsol, branchvars[i]);
      if( (boundtypes[i] == SCIP_BOUNDTYPE_LOWER && optval < branchbounds[i]) ||
          (boundtypes[i] == SCIP_BOUNDTYPE_UPPER && optval > branchbounds[i]) )
      {
         isoptimal = FALSE;
         break;
      }
   }

   /* free all local memory */
   SCIPfreeBufferArray(scip, &branchvars);
   SCIPfreeBufferArray(scip, &boundtypes);
   SCIPfreeBufferArray(scip, &branchbounds);

   if( isoptimal )
      SCIPnodeSetOptimal(node);

   return SCIP_OKAY;
}

/** read the optimal solution (modified from readSol in reader_sol.c -- don't connect the solution with primal solutions) */
/** TODO: currently the read objective is wrong, since it doesn not include objetives from multi-aggregated variables */
SCIP_RETCODE SCIPreadOptSol(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           fname,              /**< name of the input file */
   SCIP_SOL**            sol                 /**< pointer to store the solution */
   )
{
   SCIP_FILE* file;
   SCIP_Bool error;
   SCIP_Bool unknownvariablemessage;
   SCIP_Bool usevartable;
   int lineno;
   // fname = "/home/xuliming/daggerSpace/training_files/scip-dagger/bfs_solution/cauctions/test_200_1000/instance_30.sol";
   // fname = "/home/xuliming/daggerSpace/training_files/scip-dagger/bfs_solution/setcover/test_1000r_1000c_0.05d/instance_1.sol";
   // fname = "/home/xuliming/daggerSpace/training_files/scip-dagger/bfs_solution/indset/new_transfer_1000_4/instance_102.sol";
   fname = "/home/xuliming/daggerSpace/training_files/scip-dagger/bfs_solution/facilities/transfer_400_100_5/instance_87.sol";

   assert(scip != NULL);
   assert(fname != NULL);

   SCIP_CALL( SCIPgetBoolParam(scip, "misc/usevartable", &usevartable) );

   if( !usevartable )
   {
      SCIPerrorMessage("Cannot read solution file if vartable is disabled. Make sure parameter 'misc/usevartable' is set to TRUE.\n");
      return SCIP_READERROR;
   }

   /* open input file */
   file = SCIPfopen(fname, "r");
   if( file == NULL )
   {
      SCIPerrorMessage("cannot open file <%s> for reading\n", fname);
      SCIPprintSysError(fname);
      return SCIP_NOFILE;
   }

   /* create zero solution */
   SCIP_CALL( SCIPcreateSolSelf(scip, sol, NULL) );
   assert(SCIPsolIsOriginal(*sol) == TRUE);

   /* read the file */
   error = FALSE;
   unknownvariablemessage = FALSE;
   lineno = 0;
   while( !SCIPfeof(file) && !error )
   {
      char buffer[SCIP_MAXSTRLEN];
      char varname[SCIP_MAXSTRLEN];
      char valuestring[SCIP_MAXSTRLEN];
      char objstring[SCIP_MAXSTRLEN];
      SCIP_VAR* var;
      SCIP_Real value;
      int nread;

      /* get next line */
      if( SCIPfgets(buffer, (int) sizeof(buffer), file) == NULL )
         break;
      lineno++;

      /* there are some lines which may preceed the solution information */
      if( strncasecmp(buffer, "solution status:", 16) == 0 || strncasecmp(buffer, "objective value:", 16) == 0 ||
         strncasecmp(buffer, "Log started", 11) == 0 || strncasecmp(buffer, "Variable Name", 13) == 0 ||
         strncasecmp(buffer, "All other variables", 19) == 0 || strncasecmp(buffer, "\n", 1) == 0 ||
         strncasecmp(buffer, "NAME", 4) == 0 || strncasecmp(buffer, "ENDATA", 6) == 0 )    /* allow parsing of SOL-format on the MIPLIB 2003 pages */
         continue;

      /* parse the line */
      nread = sscanf(buffer, "%s %s %s\n", varname, valuestring, objstring);
      if( nread < 2 )
      {
         SCIPerrorMessage("Invalid input line %d in solution file <%s>: <%s>.\n", lineno, fname, buffer);
         error = TRUE;
         break;
      }

      /* find the variable */
      var = SCIPfindVar(scip, varname);
      if( var == NULL )
      {
         if( !unknownvariablemessage )
         {
            SCIPverbMessage(scip, SCIP_VERBLEVEL_NORMAL, NULL, "unknown variable <%s> in line %d of solution file <%s>\n",
               varname, lineno, fname);
            SCIPverbMessage(scip, SCIP_VERBLEVEL_NORMAL, NULL, "  (further unknown variables are ignored)\n");
            unknownvariablemessage = TRUE;
         }
         continue;
      }

      /* cast the value */
      if( strncasecmp(valuestring, "inv", 3) == 0 )
         continue;
      else if( strncasecmp(valuestring, "+inf", 4) == 0 || strncasecmp(valuestring, "inf", 3) == 0 )
         value = SCIPinfinity(scip);
      else if( strncasecmp(valuestring, "-inf", 4) == 0 )
         value = -SCIPinfinity(scip);
      else
      {
         nread = sscanf(valuestring, "%lf", &value);
         if( nread != 1 )
         {
            SCIPerrorMessage("Invalid solution value <%s> for variable <%s> in line %d of solution file <%s>.\n",
               valuestring, varname, lineno, fname);
            error = TRUE;
            break;
         }
      }

      /* set the solution value of the variable, if not multiaggregated */
      if( SCIPisTransformed(scip) && SCIPvarGetStatus(SCIPvarGetProbvar(var)) == SCIP_VARSTATUS_MULTAGGR )
      {
         SCIPverbMessage(scip, SCIP_VERBLEVEL_NORMAL, NULL, "ignored solution value for multiaggregated variable <%s>\n", SCIPvarGetName(var));
      }
      else
      {
         SCIP_RETCODE retcode;
         retcode =  SCIPsetSolVal(scip, *sol, var, value);

         if( retcode == SCIP_INVALIDDATA )
         {
            if( SCIPvarGetStatus(SCIPvarGetProbvar(var)) == SCIP_VARSTATUS_FIXED )
            {
               SCIPverbMessage(scip, SCIP_VERBLEVEL_NORMAL, NULL, "ignored conflicting solution value for fixed variable <%s>\n",
                  SCIPvarGetName(var));
            }
            else
            {
               SCIPverbMessage(scip, SCIP_VERBLEVEL_NORMAL, NULL, "ignored solution value for multiaggregated variable <%s>\n",
                  SCIPvarGetName(var));
            }
         }
         else
         {
            SCIP_CALL( retcode );
         }
      }
   }

   /* close input file */
   SCIPfclose(file);

   if( !error )
   {
      SCIP_Real obj;
      /* display result */
      obj = SCIPgetSolOrigObj(scip, *sol);
      SCIPverbMessage(scip, SCIP_VERBLEVEL_NORMAL, NULL, "optimal solution from solution file <%s> was %s\n",
         fname, "read");
      SCIPverbMessage(scip, SCIP_VERBLEVEL_NORMAL, NULL, "objective: %f\n", obj);
      return SCIP_OKAY;
   }
   else
   {
      /* free solution */
      SCIP_CALL( SCIPfreeSolSelf(scip, sol) );

      return SCIP_READERROR;
   }
}

/*
 * Callback methods of node selector
 */

/** solving process initialization method of node selector (called when branch and bound process is about to begin) */
static
SCIP_DECL_NODESELINIT(nodeselInitOracle)
{
   SCIP_NODESELDATA* nodeseldata;

   assert(scip != NULL);
   assert(nodesel != NULL);

   nodeseldata = SCIPnodeselGetData(nodesel);
   assert(nodeseldata != NULL);

   /** solfname should be set before including nodeseloracle */
   assert(nodeseldata->solfname != NULL);
   nodeseldata->optsol = NULL;
   SCIP_CALL( SCIPreadOptSol(scip, nodeseldata->solfname, &nodeseldata->optsol) );
   assert(nodeseldata->optsol != NULL);
#ifdef SCIP_DEBUG
   SCIP_CALL( SCIPprintSol(scip, nodeseldata->optsol, NULL, FALSE) );
#endif

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
   // SCIP_CALL( SCIPhgfeatCreate(scip, &nodeseldata->optfeat, SCIP_FEATNODESEL_SIZE) );
   assert(nodeseldata->optfeat != NULL);
   SCIPfeatSetMaxDepth(nodeseldata->optfeat, SCIPgetNBinVars(scip) + SCIPgetNIntVars(scip));

#ifndef NDEBUG
   nodeseldata->optnodenumber = -1;
#endif
   nodeseldata->negate = TRUE;

   nodeseldata->obj_so_far = INT_MAX;
   nodeseldata->cur_group_idx = 0;

   return SCIP_OKAY;
}

/** deinitialization method of node selector (called before transformed problem is freed) */
static
SCIP_DECL_NODESELEXIT(nodeselExitOracle)
{
   SCIP_NODESELDATA* nodeseldata;
   assert(scip != NULL);
   assert(nodesel != NULL);

   nodeseldata = SCIPnodeselGetData(nodesel);

   assert(nodeseldata->optsol != NULL);
   SCIP_CALL( SCIPfreeSolSelf(scip, &nodeseldata->optsol) );
   nodeseldata->optsol = NULL;

   if( nodeseldata->trjfile != NULL)
   {
      fclose(nodeseldata->wfile);
      fclose(nodeseldata->trjfile);
      nodeseldata->wfile = NULL;
      nodeseldata->trjfile = NULL;
   }

   if( nodeseldata->feat != NULL )
   {
      SCIP_CALL( SCIPfeatFree(scip, &nodeseldata->feat) );
      nodeseldata->feat = NULL;
   }

   assert(nodeseldata->left_feat != NULL);
   SCIP_CALL( SCIPfeatFree(scip, &nodeseldata->left_feat) );
   
   assert(nodeseldata->right_feat != NULL);
   SCIP_CALL( SCIPfeatFree(scip, &nodeseldata->right_feat) );

   if( nodeseldata->optfeat != NULL )
   {
      SCIP_CALL( SCIPfeatFree(scip, &nodeseldata->optfeat) );
      nodeseldata->optfeat = NULL;
   }

#ifndef NDEBUG
   nodeseldata->optnodenumber = -1;
#endif
   nodeseldata->negate = TRUE;

   nodeseldata->obj_so_far = INT_MAX;
   nodeseldata->cur_group_idx = 0;

   return SCIP_OKAY;
}

/** destructor of node selector to free user data (called when SCIP is exiting) */
static
SCIP_DECL_NODESELFREE(nodeselFreeOracle)
{
   SCIP_NODESELDATA* nodeseldata;
   nodeseldata = SCIPnodeselGetData(nodesel);

   assert(nodeseldata != NULL);

   SCIPfreeBlockMemory(scip, &nodeseldata);

   SCIPnodeselSetData(nodesel, NULL);

   return SCIP_OKAY;
}


/** node selection method of node selector */
static
SCIP_DECL_NODESELSELECT(nodeselSelectOracle)
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
   double rnd_poss=1.0;
   int rnd;
   SCIP_Bool has_opt = 0;

   assert(nodesel != NULL);
   assert(strcmp(SCIPnodeselGetName(nodesel), NODESEL_NAME) == 0);
   assert(scip != NULL);
   assert(selnode != NULL);

   nodeseldata = SCIPnodeselGetData(nodesel);
   assert(nodeseldata != NULL);

   /* collect leaves, children and siblings data */
   SCIP_CALL( SCIPgetOpenNodesData(scip, &leaves, &children, &siblings, &nleaves, &nchildren, &nsiblings) );

   optchild = -1;
   for( i = 0; i < nchildren; i++)
   {
      /* it can happen that the same children have been opt-checked already if this callback is called multiple times at
       * the same node, e.g., if an afternode or afterplunge heuristic found a solution that may have cut off the previously
       * selected node
       */
      if( ! SCIPnodeIsOptchecked(children[i]) )
      {
         SCIPnodeCheckOptimal(scip, children[i], nodeseldata->optsol);
         SCIPnodeSetOptchecked(children[i]);
      }
      
      if( SCIPnodeIsOptimal(children[i]) )
      {
         // SCIPdebugMessage("opt node #%"SCIP_LONGINT_FORMAT"\n", SCIPnodeGetNumber(children[i]));
#ifndef NDEBUG
         nodeseldata->optnodenumber = SCIPnodeGetNumber(children[i]);
#endif
         optchild = i;
      }
      // if (SCIPnodeGetNumber(children[i]) == 2) 
      //    SCIPcalcNodeselFeat(scip, children[i], nodeseldata->left_feat);
      // if (SCIPnodeGetNumber(children[i]) == 3) 
      //    SCIPcalcNodeselFeat(scip, children[i], nodeseldata->right_feat);
   }

   /* write examples - origin */
   if (FALSE)
   {
      /* point */
      if( nodeseldata->trjfile != NULL )
      {
         SCIPdebugMessage("node selection feature\n");
         if( optchild != -1 )
         {
            /* new optimal node */
            SCIPcalcNodeselFeat(scip, children[optchild], nodeseldata->optfeat);
            for( i = 0; i < nchildren; i++)
            {
               if( i != optchild )
               {
                  SCIPcalcNodeselFeat(scip, children[i], nodeseldata->feat);
                  nodeseldata->negate ^= 1;
                  SCIPfeatDiffNNPrint(scip, nodeseldata->trjfile, nodeseldata->wfile, nodeseldata->optfeat, nodeseldata->feat, 1, nodeseldata->negate);
               }
            }
            for( i = 0; i < nsiblings; i++ )
            {
               SCIPcalcNodeselFeat(scip, siblings[i], nodeseldata->feat);
               nodeseldata->negate ^= 1;
               SCIPfeatDiffNNPrint(scip, nodeseldata->trjfile, nodeseldata->wfile, nodeseldata->optfeat, nodeseldata->feat, 1, nodeseldata->negate);
            }
            for( i = 0; i < nleaves; i++ )
            {
               SCIPcalcNodeselFeat(scip, leaves[i], nodeseldata->feat);
               nodeseldata->negate ^= 1;
               SCIPfeatDiffNNPrint(scip, nodeseldata->trjfile, nodeseldata->wfile, nodeseldata->optfeat, nodeseldata->feat, 1, nodeseldata->negate);
            }
         }
         else
         {
            /* children are not optimal */
            assert(nchildren == 0 || (nchildren > 0 && nodeseldata->optnodenumber != -1));
            for( i = 0; i < nchildren; i++ )
            {
               SCIPcalcNodeselFeat(scip, children[i], nodeseldata->feat);
               nodeseldata->negate ^= 1;
               SCIPfeatDiffNNPrint(scip, nodeseldata->trjfile, nodeseldata->wfile, nodeseldata->optfeat, nodeseldata->feat, -1, nodeseldata->negate);
            }
         }
      }
   }
   else if (TRUE)
   {
      /* single feat */
      if( nodeseldata->trjfile != NULL && nchildren + nsiblings + nleaves > 1)
      {
         SCIPdebugMessage("node selection single feature\n");
         nodeseldata->cur_group_idx += 1;

         for( i = 0; i < nchildren; i++)
         {
            SCIPcalcNodeselFeat(scip, children[i], nodeseldata->feat);
            SCIPfeatSingleNNPrint(scip, nodeseldata->trjfile, nodeseldata->feat, SCIPnodeGetNumber(children[i]), nodeseldata->cur_group_idx);
         }
         for( i = 0; i < nsiblings; i++ )
         {
            SCIPcalcNodeselFeat(scip, siblings[i], nodeseldata->feat);
            SCIPfeatSingleNNPrint(scip, nodeseldata->trjfile, nodeseldata->feat, SCIPnodeGetNumber(siblings[i]), nodeseldata->cur_group_idx);
            // SCIPfeatDiffNNPrint(scip, nodeseldata->trjfile, nodeseldata->wfile, nodeseldata->optfeat, nodeseldata->feat, 1, nodeseldata->negate);
         }
         for( i = 0; i < nleaves; i++ )
         {
            SCIPcalcNodeselFeat(scip, leaves[i], nodeseldata->feat);
            SCIPfeatSingleNNPrint(scip, nodeseldata->trjfile, nodeseldata->feat, SCIPnodeGetNumber(leaves[i]), nodeseldata->cur_group_idx);
            // SCIPfeatDiffNNPrint(scip, nodeseldata->trjfile, nodeseldata->wfile, nodeseldata->optfeat, nodeseldata->feat, 1, nodeseldata->negate);
         }
      }
   }
   else
   {
      /* pair-wise write example */
      for( i = 0; i < nchildren; i++)
      {
         has_opt = SCIPnodeIsOptimal(children[i]) ? 1 : has_opt;
      }
      for ( i = 0; i < nsiblings; i++)
      {
         has_opt = SCIPnodeIsOptimal(siblings[i]) ? 1 : has_opt;
      }
      for (i = 0; i < nleaves; i++)
      {
         has_opt = SCIPnodeIsOptimal(leaves[i]) ? 1 : has_opt;
      }
      if (has_opt && nodeseldata->trjfile != NULL )
      {
         if( optchild != -1 )
         {
            /* new optimal node */
            SCIPcalcNodeselFeat(scip, children[optchild], nodeseldata->optfeat);
            for( i = 0; i < nchildren; i++)
            {
               if( i != optchild )
               {
                  SCIPcalcNodeselFeat(scip, children[i], nodeseldata->feat);
                  nodeseldata->negate ^= 1;
                  SCIPfeatDiffNNPrint(scip, nodeseldata->trjfile, nodeseldata->wfile, nodeseldata->optfeat, nodeseldata->feat, 1, nodeseldata->negate);
                  // SCIPfeatDiffNNPrintConcat(scip, nodeseldata->trjfile, nodeseldata->wfile, nodeseldata->optfeat, nodeseldata->feat, nodeseldata->left_feat, nodeseldata->right_feat, 1, nodeseldata->negate);
               }
            }
            for( i = 0; i < nsiblings; i++ )
            {
               SCIPcalcNodeselFeat(scip, siblings[i], nodeseldata->feat);
               nodeseldata->negate ^= 1;
               SCIPfeatDiffNNPrint(scip, nodeseldata->trjfile, nodeseldata->wfile, nodeseldata->optfeat, nodeseldata->feat, 1, nodeseldata->negate);
               // SCIPfeatDiffNNPrintConcat(scip, nodeseldata->trjfile, nodeseldata->wfile, nodeseldata->optfeat, nodeseldata->feat, nodeseldata->left_feat, nodeseldata->right_feat, 1, nodeseldata->negate);
            }
            for( i = 0; i < nleaves; i++ )
            {
               SCIPcalcNodeselFeat(scip, leaves[i], nodeseldata->feat);
               nodeseldata->negate ^= 1;
               SCIPfeatDiffNNPrint(scip, nodeseldata->trjfile, nodeseldata->wfile, nodeseldata->optfeat, nodeseldata->feat, 1, nodeseldata->negate);
               // SCIPfeatDiffNNPrintConcat(scip, nodeseldata->trjfile, nodeseldata->wfile, nodeseldata->optfeat, nodeseldata->feat, nodeseldata->left_feat, nodeseldata->right_feat, 1, nodeseldata->negate);
            }
         }
         else
         {
            /* children are not optimal */
            assert(nchildren == 0 || (nchildren > 0 && nodeseldata->optnodenumber != -1));
            for( i = 0; i < nchildren; i++ )
            {
               SCIPcalcNodeselFeat(scip, children[i], nodeseldata->feat);
               nodeseldata->negate ^= 1;
               SCIPfeatDiffNNPrint(scip, nodeseldata->trjfile, nodeseldata->wfile, nodeseldata->optfeat, nodeseldata->feat, 11, nodeseldata->negate);
               // SCIPfeatDiffNNPrintConcat(scip, nodeseldata->trjfile, nodeseldata->wfile, nodeseldata->optfeat, nodeseldata->feat, nodeseldata->left_feat, nodeseldata->right_feat, 1, nodeseldata->negate);
            }
         }
      }
      else if ( ! has_opt && nodeseldata->trjfile != NULL )
      {
         /* children are not optimal */
         assert(nchildren == 0 || (nchildren > 0 && nodeseldata->optnodenumber != -1));
         for( i = 0; i < nchildren; i++ )
         {
            SCIPcalcNodeselFeat(scip, children[i], nodeseldata->feat);
            nodeseldata->negate ^= 1;
            SCIPfeatDiffNNPrint(scip, nodeseldata->trjfile, nodeseldata->wfile, nodeseldata->optfeat, nodeseldata->feat, -1, nodeseldata->negate);
            // SCIPfeatDiffNNPrintConcat(scip, nodeseldata->trjfile, nodeseldata->wfile, nodeseldata->optfeat, nodeseldata->feat, nodeseldata->left_feat, nodeseldata->right_feat, -1, nodeseldata->negate);
         }
      }
   }

   if (FALSE)
   {
      /* random possibility */
      rnd = rand();

      for( i = 0; i < nchildren; i++)
      {
         has_opt = SCIPnodeIsOptimal(children[i]) ? 1 : has_opt;
      }
      for ( i = 0; i < nsiblings; i++)
      {
         has_opt = SCIPnodeIsOptimal(siblings[i]) ? 1 : has_opt;
      }
      for (i = 0; i < nleaves; i++)
      {
         has_opt = SCIPnodeIsOptimal(leaves[i]) ? 1 : has_opt;
      }

      /* best or worst: draw oracle graph */
      if ( has_opt && rnd >= rnd_poss * RAND_MAX )
      {
         // *selnode = SCIPgetWorstNode(scip);
         SCIPdebugMessage("maxrnd\n");
      }
      else
      {
         *selnode = SCIPgetBestNode(scip);
         SCIPdebugMessage("minrnd\n");
      }
   }
   
   else
   {
      *selnode = SCIPgetBestNode(scip);
   }
   
   /* experiment start: ml-HeFeatures */

   if (TRUE)
   {
      int idx = -1;
      int isopt = -1;
      int depth = -1;
      int left = nchildren + nsiblings + nleaves;
      
      SCIP_Real lowerbound = -1;
      SCIP_Real dualbound = SCIPgetDualbound(scip);
      SCIP_Real primalbound = SCIPgetPrimalbound(scip);
      
      int nsols = SCIPgetNSols(scip);
      int s;
      
      //time_t time_ptr;
      SCIP_Real bPB_time = SCIPgetSolTime(scip, SCIPgetBestSol(scip));
      SCIP_Real current_time = SCIPgetSolvingTime(scip);
      SCIP_Real total_time = SCIPgetTotalTime(scip);

      int stage = -1;
      stage = SCIPgetStage(scip);

      if (*selnode != NULL)
      {
         SCIPprintNodeRootPath(scip, *selnode, NULL);        
         idx = SCIPnodeGetNumber(*selnode);
        
         if (SCIPnodeIsOptimal(*selnode))
            isopt = 1;
        
         depth = SCIPnodeGetDepth(*selnode);
         lowerbound = SCIPnodeGetLowerbound(*selnode);
      }
      //time(&time_ptr);

      SCIPdebugMsg(scip, "final selecting node number %d primalbound %.3f lowerbound %.3f dualbound %.3f time %.2f depth %d left %d bPB_time %.2f opt %d nsols %d ", 
                                                   idx, primalbound, lowerbound, dualbound, current_time, depth, left, bPB_time, isopt, nsols);
      
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
         SCIP_SOL** sols;
         sols = SCIPgetSols(scip);

         for (s = 0; s < nsols && s < 10; ++s)
         {
            SCIP_SOL* sol;
            sol = sols[s];
            SCIPprintSol(scip, sol, NULL, FALSE);
         }
         
         SCIPdebugMsg(scip, "SCIPgetSolNodenum %lld, SCIPgetNBestSolsFound %lld\n", SCIPgetSolNodenum(scip, SCIPgetBestSol(scip)), SCIPgetNBestSolsFound(scip));
      }

   }

   if (FALSE)
      SCIPdebugMsg(scip, "Selecting node number %lld\n", *selnode != NULL ? SCIPnodeGetNumber(*selnode) : -1l);

   // // SCIPgetSolNodenum(scip, SCIPgetBestSol(scip));
   
   // // SCIPgetBestboundNode()
   // if (SCIPisGT(scip, nodeseldata->obj_so_far, SCIPgetSolOrigObj(scip, SCIPgetBestSol(scip))))
   // {
   //    nodeseldata->obj_so_far = SCIPgetSolOrigObj(scip, SCIPgetBestSol(scip));
   //    SCIPprintSol(scip, SCIPgetBestSol(scip),NULL, FALSE);
   // }

   // SCIPdebugMessage("objval %.3f, in run %d, after %"SCIP_LONGINT_FORMAT" nodes, %.3f seconds, depth %d, found by <%s>\n",
   //                   SCIPgetSolOrigObj(scip, SCIPgetBestSol(scip)),
   //                   SCIPgetSolRunnum(scip, SCIPgetBestSol(scip)),
   //                   SCIPgetSolNodenum(scip, SCIPgetBestSol(scip)),
   //                   SCIPgetSolTime(scip, SCIPgetBestSol(scip)),
   //                   SCIPgetDepth(scip),
   //                   SCIPgetSolHeur(scip, SCIPgetBestSol(scip)) != NULL
   //                   ? SCIPheurGetName(SCIPgetSolHeur(scip, SCIPgetBestSol(scip)))
   //                   : SCIPgetSolRunnum(scip, SCIPgetBestSol(scip)) == 0 ? "initial" : "relaxation"
   //                   );

   /* experiment end */

   return SCIP_OKAY;
}

/** node comparison method of oracle node selector */
static
SCIP_DECL_NODESELCOMP(nodeselCompOracle)
{  /*lint --e{715}*/
   SCIP_Bool isoptimal1 = FALSE;
   SCIP_Bool isoptimal2 = FALSE;

   int node1_idx = SCIPnodeGetNumber(node1);
   int node2_idx = SCIPnodeGetNumber(node2);

   assert(nodesel != NULL);
   assert(strcmp(SCIPnodeselGetName(nodesel), NODESEL_NAME) == 0);
   assert(scip != NULL);

   assert(SCIPnodeIsOptchecked(node1) == TRUE);
   assert(SCIPnodeIsOptchecked(node2) == TRUE);

   isoptimal1 = SCIPnodeIsOptimal(node1);
   isoptimal2 = SCIPnodeIsOptimal(node2);

   // /* 根节点的两个子节点永远优于优先队列中其余节点 */
   if (node1_idx <= 3 && node2_idx > 3)
      return -1;
   else if (node2_idx <= 3 && node1_idx > 3)
      return 1;


   if( isoptimal1 == TRUE )
   {
      assert(isoptimal2 != TRUE);
      return -1;
   }
   else if( isoptimal2 == TRUE )
      return +1;
   else
   {
      /////////////////////////////////////////////////////////////////////////////
      if (TRUE)
      {
         SCIP_Real lowerbound1;
         SCIP_Real lowerbound2;

         assert(nodesel != NULL);
         assert(strcmp(SCIPnodeselGetName(nodesel), NODESEL_NAME) == 0);
         assert(scip != NULL);

         lowerbound1 = SCIPnodeGetLowerbound(node1);
         lowerbound2 = SCIPnodeGetLowerbound(node2);
         if( SCIPisLT(scip, lowerbound1, lowerbound2) )
            return -1;
         else if( SCIPisGT(scip, lowerbound1, lowerbound2) )
            return +1;
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
                  return -1;
               else if( nodetype1 != SCIP_NODETYPE_CHILD && nodetype2 == SCIP_NODETYPE_CHILD )
                  return +1;
               else if( nodetype1 == SCIP_NODETYPE_SIBLING && nodetype2 != SCIP_NODETYPE_SIBLING )
                  return -1;
               else if( nodetype1 != SCIP_NODETYPE_SIBLING && nodetype2 == SCIP_NODETYPE_SIBLING )
                  return +1;
               else
               {
                  int depth1;
                  int depth2;

                  depth1 = SCIPnodeGetDepth(node1);
                  depth2 = SCIPnodeGetDepth(node2);
                  if( depth1 < depth2 )
                     return -1;
                  else if( depth1 > depth2 )
                     return +1;
                  else
                     return 0;
               }
            }

            if( SCIPisLT(scip, estimate1, estimate2) )
               return -1;

            assert(SCIPisGT(scip, estimate1, estimate2));
            return +1;
         }
      }
      else
      {
         int depth1;
         int depth2;
   
         depth1 = SCIPnodeGetDepth(node1);
         depth2 = SCIPnodeGetDepth(node2);
         if( depth1 > depth2 )
            return -1;
         else if( depth1 < depth2 )
            return +1;
         else
         {
            SCIP_Real lowerbound1;
            SCIP_Real lowerbound2;
   
            lowerbound1 = SCIPnodeGetLowerbound(node1);
            lowerbound2 = SCIPnodeGetLowerbound(node2);
            if( SCIPisLT(scip, lowerbound1, lowerbound2) )
               return -1;
            else if( SCIPisGT(scip, lowerbound1, lowerbound2) )
               return +1;
            if( lowerbound1 < lowerbound2 )
               return -1;
            else if( lowerbound1 > lowerbound2 )
               return +1;
            else
               return 0;
         }
      }
   }
}

/*
 * node selector specific interface methods
 */

/** creates the uct node selector and includes it in SCIP */
SCIP_RETCODE SCIPincludeNodeselOracle(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_NODESELDATA* nodeseldata;
   SCIP_NODESEL* nodesel;

   /* create oracle node selector data */
   SCIP_CALL( SCIPallocBlockMemory(scip, &nodeseldata) );

   nodesel = NULL;

   /* use SCIPincludeNodeselBasic() plus setter functions if you want to set callbacks one-by-one and your code should
    * compile independent of new callbacks being added in future SCIP versions
    */
   SCIP_CALL( SCIPincludeNodeselBasic(scip, &nodesel, NODESEL_NAME, NODESEL_DESC, NODESEL_STDPRIORITY,
          NODESEL_MEMSAVEPRIORITY, nodeselSelectOracle, nodeselCompOracle, nodeseldata) );

   assert(nodesel != NULL);

   /* set non fundamental callbacks via setter functions */
   SCIP_CALL( SCIPsetNodeselCopy(scip, nodesel, NULL) );
   SCIP_CALL( SCIPsetNodeselInit(scip, nodesel, nodeselInitOracle) );
   SCIP_CALL( SCIPsetNodeselExit(scip, nodesel, nodeselExitOracle) );
   SCIP_CALL( SCIPsetNodeselFree(scip, nodesel, nodeselFreeOracle) );

   /* add oracle node selector parameters */
   nodeseldata->solfname = NULL;
   SCIP_CALL( SCIPaddStringParam(scip,
         "nodeselection/"NODESEL_NAME"/solfname",
         "name of the optimal solution file",
         &nodeseldata->solfname, TRUE, DEFAULT_FILENAME, NULL, NULL) );
   nodeseldata->trjfname = NULL;
   SCIP_CALL( SCIPaddStringParam(scip,
         "nodeselection/"NODESEL_NAME"/trjfname",
         "name of the file to write node selection trajectories",
         &nodeseldata->trjfname, TRUE, DEFAULT_FILENAME, NULL, NULL) );

   return SCIP_OKAY;
}

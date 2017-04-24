#pragma once

#include <functional>

#include <malloc.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

template<typename REAL,
		 typename OBJECTIVE_FUNC_TYPE = std::function<REAL(REAL*)>,
         typename CONTRAIN_FUNC_TYPE = std::function<void(REAL*,int)>,
         typename VERBOSE_FUNC_TYPE = std::function<void(const char*)> >
class NMSimplexT
{
public:
    unsigned int MAX_IT;
    REAL ALPHA;
    REAL BETA;
    REAL GAMMA;
    REAL EPSILON;
    NMSimplexT() :
        MAX_IT(1000),
        ALPHA(1.0),
        BETA(0.5),
        GAMMA(2.0),
        EPSILON(1e-4),
        scale(1.0)
    {

    }
    //typedef REAL(*OBJECTIVE_FUNC_TYPE)(REAL*);
    //typedef void(*CONSTRAIN_FUNC_TYPE)(REAL*, int n);

    OBJECTIVE_FUNC_TYPE objfunc;
    CONTRAIN_FUNC_TYPE constrain;
    VERBOSE_FUNC_TYPE verbose;
    

    REAL scale;
    
    REAL Optimize(REAL* start, int n)
    {

        int vs;         /* vertex with smallest value */
        int vh;         /* vertex with next smallest value */
        int vg;         /* vertex with largest value */

        int i, j, m, row;
        int k;   	      /* track the number of function evaluations */
        int itr;	      /* track the number of iterations */

        REAL **v;     /* holds vertices of simplex */
        REAL pn, qn;   /* values used to create initial simplex */
        REAL *f;      /* value of function at each vertex */
        REAL fr;      /* value of function at reflection point */
        REAL fe;      /* value of function at expansion point */
        REAL fc;      /* value of function at contraction point */
        REAL *vr;     /* reflection - coordinates */
        REAL *ve;     /* expansion - coordinates */
        REAL *vc;     /* contraction - coordinates */
        REAL *vm;     /* centroid - coordinates */
        REAL min;

        REAL fsum, favg, s, cent;

        char str_buff[100];
        /* dynamically allocate arrays */

        /* allocate the rows of the arrays */
        v = (REAL **)malloc((n + 1) * sizeof(REAL *));
        f = (REAL *)malloc((n + 1) * sizeof(REAL));
        vr = (REAL *)malloc(n * sizeof(REAL));
        ve = (REAL *)malloc(n * sizeof(REAL));
        vc = (REAL *)malloc(n * sizeof(REAL));
        vm = (REAL *)malloc(n * sizeof(REAL));

        /* allocate the columns of the arrays */
        for (i = 0; i <= n; i++) {
            v[i] = (REAL *)malloc(n * sizeof(REAL));
        }

        /* create the initial simplex */
        /* assume one of the vertices is 0,0 */

        pn = scale*(sqrt(n + 1) - 1 + n) / (n*sqrt(2));
        qn = scale*(sqrt(n + 1) - 1) / (n*sqrt(2));

        for (i = 0; i<n; i++) {
            v[0][i] = start[i];
        }

        for (i = 1; i <= n; i++) {
            for (j = 0; j<n; j++) {
                if (i - 1 == j) {
                    v[i][j] = pn + start[j];
                }
                else {
                    v[i][j] = qn + start[j];
                }
            }

			if (constrain != NULL) {
				constrain(v[i], n);
			}
        }

        /* find the initial function values */
        for (j = 0; j <= n; j++) {
            f[j] = objfunc(v[j]);
        }

        k = n + 1;

        /* print out the initial values */
        if (verbose)
        {

            sprintf_s(str_buff, "Initial Values\n");
            verbose(str_buff);
            for (j = 0; j <= n; j++) {
                for (i = 0; i < n; i++) {
                    sprintf_s(str_buff, "%f %f\n", v[j][i], f[j]);
                    verbose(str_buff);
                }
            }
        }

        /* begin the main loop of the minimization */
        for (itr = 1; itr <= MAX_IT; itr++) {
            /* find the index of the largest value */
            vg = 0;
            for (j = 0; j <= n; j++) {
                if (f[j] > f[vg]) {
                    vg = j;
                }
            }

            /* find the index of the smallest value */
            vs = 0;
            for (j = 0; j <= n; j++) {
                if (f[j] < f[vs]) {
                    vs = j;
                }
            }

            /* find the index of the second largest value */
            vh = vs;
            for (j = 0; j <= n; j++) {
                if (f[j] > f[vh] && f[j] < f[vg]) {
                    vh = j;
                }
            }

            /* calculate the centroid */
            for (j = 0; j <= n - 1; j++) {
                cent = 0.0;
                for (m = 0; m <= n; m++) {
                    if (m != vg) {
                        cent += v[m][j];
                    }
                }
                vm[j] = cent / n;
            }

            /* reflect vg to new vertex vr */
            for (j = 0; j <= n - 1; j++) {
                /*vr[j] = (1+ALPHA)*vm[j] - ALPHA*v[vg][j];*/
                vr[j] = vm[j] + ALPHA*(vm[j] - v[vg][j]);
            }
            if (constrain != NULL) {
                constrain(vr, n);
            }
            fr = objfunc(vr);
            k++;

            if (fr < f[vh] && fr >= f[vs]) {
                for (j = 0; j <= n - 1; j++) {
                    v[vg][j] = vr[j];
                }
                f[vg] = fr;
            }

            /* investigate a step further in this direction */
            if (fr <  f[vs]) {
                for (j = 0; j <= n - 1; j++) {
                    /*ve[j] = GAMMA*vr[j] + (1-GAMMA)*vm[j];*/
                    ve[j] = vm[j] + GAMMA*(vr[j] - vm[j]);
                }
                if (constrain != NULL) {
                    constrain(ve, n);
                }
                fe = objfunc(ve);
                k++;

                /* by making fe < fr as opposed to fe < f[vs],
                Rosenbrocks function takes 63 iterations as opposed
                to 64 when using REAL variables. */

                if (fe < fr) {
                    for (j = 0; j <= n - 1; j++) {
                        v[vg][j] = ve[j];
                    }
                    f[vg] = fe;
                }
                else {
                    for (j = 0; j <= n - 1; j++) {
                        v[vg][j] = vr[j];
                    }
                    f[vg] = fr;
                }
            }

            /* check to see if a contraction is necessary */
            if (fr >= f[vh]) {
                if (fr < f[vg] && fr >= f[vh]) {
                    /* perform outside contraction */
                    for (j = 0; j <= n - 1; j++) {
                        /*vc[j] = BETA*v[vg][j] + (1-BETA)*vm[j];*/
                        vc[j] = vm[j] + BETA*(vr[j] - vm[j]);
                    }
                    if (constrain != NULL) {
                        constrain(vc, n);
                    }
                    fc = objfunc(vc);
                    k++;
                }
                else {
                    /* perform inside contraction */
                    for (j = 0; j <= n - 1; j++) {
                        /*vc[j] = BETA*v[vg][j] + (1-BETA)*vm[j];*/
                        vc[j] = vm[j] - BETA*(vm[j] - v[vg][j]);
                    }
                    if (constrain != NULL) {
                        constrain(vc, n);
                    }
                    fc = objfunc(vc);
                    k++;
                }


                if (fc < f[vg]) {
                    for (j = 0; j <= n - 1; j++) {
                        v[vg][j] = vc[j];
                    }
                    f[vg] = fc;
                }
                /* at this point the contraction is not successful,
                we must halve the distance from vs to all the
                vertices of the simplex and then continue.
                10/31/97 - modified to account for ALL vertices.
                */
                else {
                    for (row = 0; row <= n; row++) {
                        if (row != vs) {
                            for (j = 0; j <= n - 1; j++) {
                                v[row][j] = v[vs][j] + (v[row][j] - v[vs][j]) / 2.0;
                            }
                        }
                    }
                    if (constrain != NULL) {
                        constrain(v[vg], n);
                    }
                    f[vg] = objfunc(v[vg]);
                    k++;
                    if (constrain != NULL) {
                        constrain(v[vh], n);
                    }
                    f[vh] = objfunc(v[vh]);
                    k++;


                }
            }

            /* print out the value at each iteration */
            if (verbose)
            {
                sprintf_s(str_buff, "Iteration %d\n", itr);
                verbose(str_buff);
                for (j = 0; j <= n; j++) {
                    for (i = 0; i < n; i++) {
                        sprintf_s(str_buff, "%f %f\n", v[j][i], f[j]);
                        verbose(str_buff);
                    }
                }
            }
            /* test for convergence */
            fsum = 0.0;
            for (j = 0; j <= n; j++) {
                fsum += f[j];
            }
            favg = fsum / (n + 1);
            s = 0.0;
            for (j = 0; j <= n; j++) {
                s += pow((f[j] - favg), 2.0) / (n);
            }
            s = sqrt(s);
            if (s < EPSILON) break;
        }
        /* end main loop of the minimization */

        /* find the index of the smallest value */
        vs = 0;
        for (j = 0; j <= n; j++) {
            if (f[j] < f[vs]) {
                vs = j;
            }
        }
        if (verbose)
        {

            sprintf_s(str_buff, "The minimum was found at\n");
            verbose(str_buff);
            for (j = 0; j < n; j++) {
                sprintf_s(str_buff, "%e\n", v[vs][j]);
                verbose(str_buff);
            }
        }
        for (j = 0; j < n; j++) start[j] = v[vs][j];

        min = objfunc(v[vs]);
        k++;
        if (verbose)
        {

            sprintf_s(str_buff, "%d Function Evaluations\n", k);
            verbose(str_buff);
            sprintf_s(str_buff, "%d Iterations through program\n", itr);
            verbose(str_buff);
        }

        free(f);
        free(vr);
        free(ve);
        free(vc);
        free(vm);
        for (i = 0; i <= n; i++) {
            free(v[i]);
        }
        free(v);
        return min;
    }






};

typedef NMSimplexT<double> NMSimplex;
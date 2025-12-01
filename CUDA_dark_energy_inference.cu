

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<limits.h>
#include "random.c"
#define num1h		32        
#define num		1048
#define num1		3
#define num2		3
#define nsteps		100000
#define nsteps1		100000
#define mp		1.0f 
#define omg_r0		9.0f*(pow(10.0f, -5.0f) )
#define  c              3.0f*(pow(10.0f, 5.0f) )
#define rs              153.3f 
#define	H0		67.4
#define sig_H0		0.5f



   double z1[3] = {0.106f, 0.2f, 0.35f } ;
   double x1b1[num1] = {0.90415f, 0.83333f, 0.740740f } ;

   double z2[3] = { 0.44f, 0.60f, 0.73f} ;
   double x1b2[num2] = { 0.69444f, 0.625f, 0.57803f } ;

   double dz1_ob[num1][1] ; double az2_ob[num2][1];
   double sig_dz1[num1]= {0.015f, 0.0061f, 0.0036f} ;
 
   double cov_d[num1][num1] ; double cov_a[num2][num2] ;
   double sig_az2[num2]={ 0.034f, 0.020f, 0.021f } ;

   double sig_az1[num2]={ 0.028f, 0.016f, 0.016f } ;
   double az1_ob[num1] = {0.526f, 0.488f, 0.484f} ;

   double z1h[num1h];  double H_ob[num1h];
   double sig_obh[num1h] ; double x1[num1h] ;
 








__host__ __device__ double dy1(double x, double aa1, double aa2, double cc, double y1, double y2, double H_00)     
  {
     return (y2) / (x*H_00 ) ; 
  }

 __host__ __device__ double dy2(double x, double aa1, double aa2, double cc, double y1, double y2, double H_00)     
  {
      return (-3.0f*y2/x) - ( (1.0f/(x*H_00))* ( ( (pow((2.0f/3.0f),0.5f) * cc * cosh(y1/pow((6.0f*aa2),0.5f)) * pow((1.0f/(cosh(y1/pow((6.0f*aa1),0.5f)))) ,2.0f) * tanh(y1/pow((6.0f*aa1),0.5f)) )/pow(aa1,0.5f) )  +  ((cc * sinh(y1/pow((6.0f*aa2),0.5f)) * pow((tanh(y1/pow((6.0f*aa1),0.5f))),2.0f) )/pow((6.0f*aa2),0.5f)) )  )     ;        
      
      
}

__host__ __device__ double H_th(double x, double omg_m0, double aa1, double aa2, double cc, double y1, double y2, double omg_k0)             
  {
 
    return  (pow(((omg_m0/pow(x,3.0f)) + (omg_k0/pow(x,2.0f)) + (omg_r0/pow(x,4.0f)) + (y2*y2/(6.0f)) + ( (1.0f/3.0f)* ( cc * pow((tanh(y1/pow((6.0f*aa1),0.5f))),2.0f) * cosh(y1/pow((6.0f*aa2),0.5f)) )  )   ),0.5f)) ;
    
  
  }







__global__  void  supernova(double *x1s,  int nn, double omg_m0, double aa1, double aa2, double cc, double *H_theo_n, double omg_k0)
  {
  
        int ppp;
  
  	double tot11s, x_ini, y2_ini, y1_ini, H_ini, v1, h, k11, k21, k12, k22, k13, k23, k14, k24 ;  
   	__shared__   int p[( 1048*sizeof(int)) ];
   	__shared__   int jp[( 1*sizeof(int)) ];  
        __shared__   int j[( 1048*sizeof(int)) ];
       

   // int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;  

if (gid < nn) {
	
	
	
	

//x_fin   = x1s[gid] ;
x_ini	= 0.0003f ;
//h       = 0.00050f ;
h       = 0.0003f ;
//tot11b1  = 0.0f ;
y2_ini  = 0.0f ;

//v1      = (1.0f - omg_m0 )*(3.0f/2.0f)*(pow(H_0,2.0f))*(1.0f - w0)* 1.0f/cosh(l0*y2_ini)  ;

y1_ini  =  0.5f ;

H_ini   =   H_th(x_ini, omg_m0, aa1, aa2, cc, y1_ini, y2_ini, omg_k0) ;


p[gid]    = (x1s[gid] - x_ini)/h ;
	
	
      for ( j[gid] = 0; j[gid]<= p[gid]-1 ; j[gid]++)  {

       //   tot11b1 =   tot11b1  + ( H_0/((pow(x_ini,2.0f))*H_ini)) ;
             
          k11 = dy1(x_ini, aa1, aa2, cc, y1_ini, y2_ini, H_ini) ;
          k21 = dy2(x_ini, aa1, aa2, cc, y1_ini, y2_ini, H_ini) ;

          k12 = dy1(x_ini + 0.50f*h, aa1, aa2, cc, y1_ini + k11*(h/2.0f), y2_ini + k21*(h/2.0f), H_ini) ;
          k22 = dy2(x_ini + 0.50f*h, aa1, aa2, cc, y1_ini + k11*(h/2.0f), y2_ini + k21*(h/2.0f), H_ini) ;

          k13 = dy1(x_ini + 0.50f*h, aa1, aa2, cc, y1_ini + k12*(h/2.0f), y2_ini + k22*(h/2.0f), H_ini) ;
          k23 = dy2(x_ini + 0.50f*h, aa1, aa2, cc, y1_ini + k12*(h/2.0f), y2_ini + k22*(h/2.0f), H_ini) ;

          k14 = dy1(x_ini + h, aa1, aa2, cc, y1_ini + k13*h, y2_ini + k23*h, H_ini) ;
          k24 = dy2(x_ini + h, aa1, aa2, cc, y1_ini + k13*h, y2_ini + k23*h, H_ini) ;

          x_ini  = x_ini + h ;

          y1_ini = y1_ini + (k11+2.0f*k12+2.0f*k13+k14)*(h/6.0f) ;
          y2_ini = y2_ini + (k21+2.0f*k22+2.0f*k23+k24)*(h/6.0f) ;

          H_ini       =   H_th(x_ini, omg_m0, aa1, aa2, cc, y1_ini, y2_ini, omg_k0) ;

     }
	

//x_fin   = x1b1[i] ;
//x_ini	= 0.0000001 ;
//h       = 0.00050f ;
h       =  0.0003f ;
tot11s  = 0.0f ;
//y2_ini  = 0.0f ;

//v1      = (1.0f - omg_m0 )*(3.0f/2.0f)*(pow(H_0,2.0f))*(1.0f - w0)* 1.0f/cosh(l0*y2_ini)  ;

//y1_ini  = 10.0f;

H_ini   =   H_th(x_ini, omg_m0, aa1, aa2, cc, y1_ini, y2_ini, omg_k0) ;


ppp    = (1.0f - x_ini)/h ;	
	
      for ( jp[gid] = 0; jp[gid]<= ppp-1 ; jp[gid]++)  {
      	
      	
   

            tot11s =   tot11s  + ( 1.0f/((pow(x_ini,2.0f))*H_ini)) ;
            
            
            
            
             
          k11 = dy1(x_ini, aa1, aa2, cc, y1_ini, y2_ini, H_ini) ;
          k21 = dy2(x_ini, aa1, aa2, cc, y1_ini, y2_ini, H_ini) ;

          k12 = dy1(x_ini + 0.50f*h, aa1, aa2, cc, y1_ini + k11*(h/2.0f), y2_ini + k21*(h/2.0f), H_ini) ;
          k22 = dy2(x_ini + 0.50f*h, aa1, aa2, cc, y1_ini + k11*(h/2.0f), y2_ini + k21*(h/2.0f), H_ini) ;

          k13 = dy1(x_ini + 0.50f*h, aa1, aa2, cc, y1_ini + k12*(h/2.0f), y2_ini + k22*(h/2.0f), H_ini) ;
          k23 = dy2(x_ini + 0.50f*h, aa1, aa2, cc, y1_ini + k12*(h/2.0f), y2_ini + k22*(h/2.0f), H_ini) ;

          k14 = dy1(x_ini + h, aa1, aa2, cc, y1_ini + k13*h, y2_ini + k23*h, H_ini) ;
          k24 = dy2(x_ini + h, aa1, aa2, cc, y1_ini + k13*h, y2_ini + k23*h, H_ini) ;

          x_ini  = x_ini + h ;

          y1_ini = y1_ini + (k11+2.0f*k12+2.0f*k13+k14)*(h/6.0f) ;
          y2_ini = y2_ini + (k21+2.0f*k22+2.0f*k23+k24)*(h/6.0f) ;

          H_ini       =   H_th(x_ini, omg_m0, aa1, aa2, cc, y1_ini, y2_ini, omg_k0) ;

     }	
	



    H_theo_n[gid] 	 =    tot11s * h;

}
   

   }


  

  void find_l(double omg_m0, double aa1, double aa2, double cc, double *y11, double *y22, double *chi, double omg_k0)
   {

  int i, j, kk, p ;
  double H_theo[num1h] ;
  double integral_new[num], mu_th[num], tot11ss[num], tot11ssx[num] ;
  double x_ini, y1_ini, y2_ini, H_ini;
  double k11, k21, k12, k22, k13, k23, k14, k24 ;
  double  h ;
  double    sum1h,  chi_hubble ;
  double  Dv[num1];
  double tot11bb1[num1], H_theo_bao1[num1], tot11bb1x[num1] ;
  double a_th1[num1];
  double d_th[num1][1], del_d1[num1][1] ;
  double a_th2[num2][1], del_A2[num2][1] ;
  double trans_del_d1[1][num1] , trans_del_A2[1][num2] ;
  double  matmul_sum[1][1] ;
  double matmul11[num1][1] , matmul12[1][1] ;
  double matmul21[num2][1] , matmul22[1][1] ;
 double  Dv2[num2];
   double d_th2[num2];
  double  tot11bb2[num2], H_theo_bao2[num2], tot11bb2x[num2] ;
  double  tot11b1, chi_bao, tot11b2, x_fin ;
  
  double buf1, chi_bao1, curv ;
  double  sigma[num1h] ;
  


buf1 =0.0f ;
//H0 = 67.4f;
//sig_H0 = 0.5f ;

curv = -(omg_k0)*pow(H0 ,2.0f) ;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

for ( i = 0; i <= num1-1; i++)  {




x_fin   = x1b1[i] ;
x_ini	=  0.0003f ;
//h       = 0.00050f ;
h       =  0.0003f ;
//tot11b1  = 0.0f ;
y2_ini  = 0.0f ;

//v1      = (1.0f - omg_m0 )*(3.0f/2.0f)*(pow(H_0,2.0f))*(1.0f - w0)* 1.0f/cosh(l0*y2_ini)  ;

y1_ini  = 0.5f ;

H_ini   =   H_th(x_ini, omg_m0, aa1, aa2, cc, y1_ini, y2_ini, omg_k0) ;


p    = (x_fin - x_ini)/h ;




      for ( j = 0; j<= p-1 ; j++)  {

       //   tot11b1 =   tot11b1  + ( H_0/((pow(x_ini,2.0f))*H_ini)) ;
             
          k11 = dy1(x_ini, aa1, aa2, cc, y1_ini, y2_ini, H_ini) ;
          k21 = dy2(x_ini, aa1, aa2, cc, y1_ini, y2_ini, H_ini) ;

          k12 = dy1(x_ini + 0.50f*h, aa1, aa2, cc, y1_ini + k11*(h/2.0f), y2_ini + k21*(h/2.0f), H_ini) ;
          k22 = dy2(x_ini + 0.50f*h, aa1, aa2, cc, y1_ini + k11*(h/2.0f), y2_ini + k21*(h/2.0f), H_ini) ;

          k13 = dy1(x_ini + 0.50f*h, aa1, aa2, cc, y1_ini + k12*(h/2.0f), y2_ini + k22*(h/2.0f), H_ini) ;
          k23 = dy2(x_ini + 0.50f*h, aa1, aa2, cc, y1_ini + k12*(h/2.0f), y2_ini + k22*(h/2.0f), H_ini) ;

          k14 = dy1(x_ini + h, aa1, aa2, cc, y1_ini + k13*h, y2_ini + k23*h, H_ini) ;
          k24 = dy2(x_ini + h, aa1, aa2, cc, y1_ini + k13*h, y2_ini + k23*h, H_ini) ;

          x_ini  = x_ini + h ;

          y1_ini = y1_ini + (k11+2.0f*k12+2.0f*k13+k14)*(h/6.0f) ;
          y2_ini = y2_ini + (k21+2.0f*k22+2.0f*k23+k24)*(h/6.0f) ;

          H_ini       =   H_th(x_ini, omg_m0, aa1, aa2, cc, y1_ini, y2_ini, omg_k0) ;

     }


//x_fin   = x1b1[i] ;
//x_ini	= 0.0000001 ;
//h       = 0.00050f ;
h       =  0.0003f ;
tot11b1  = 0.0f ;
//y2_ini  = 0.0f ;

//v1      = (1.0f - omg_m0 )*(3.0f/2.0f)*(pow(H_0,2.0f))*(1.0f - w0)* 1.0f/cosh(l0*y2_ini)  ;

//y1_ini  = 10.0f;

H_ini   =   H_th(x_ini, omg_m0, aa1, aa2, cc, y1_ini, y2_ini, omg_k0) ;


p    = (1.0f - x_ini)/h ;




      for ( j = 0; j<= p-1 ; j++)  {

          tot11b1 =   tot11b1  + ( 1.0f/((pow(x_ini,2.0f))*H_ini)) ;
             
          k11 = dy1(x_ini, aa1, aa2, cc, y1_ini, y2_ini, H_ini) ;
          k21 = dy2(x_ini, aa1, aa2, cc, y1_ini, y2_ini, H_ini) ;

          k12 = dy1(x_ini + 0.50f*h, aa1, aa2, cc, y1_ini + k11*(h/2.0f), y2_ini + k21*(h/2.0f), H_ini) ;
          k22 = dy2(x_ini + 0.50f*h, aa1, aa2, cc, y1_ini + k11*(h/2.0f), y2_ini + k21*(h/2.0f), H_ini) ;

          k13 = dy1(x_ini + 0.50f*h, aa1, aa2, cc, y1_ini + k12*(h/2.0f), y2_ini + k22*(h/2.0f), H_ini) ;
          k23 = dy2(x_ini + 0.50f*h, aa1, aa2, cc, y1_ini + k12*(h/2.0f), y2_ini + k22*(h/2.0f), H_ini) ;

          k14 = dy1(x_ini + h, aa1, aa2, cc, y1_ini + k13*h, y2_ini + k23*h, H_ini) ;
          k24 = dy2(x_ini + h, aa1, aa2, cc, y1_ini + k13*h, y2_ini + k23*h, H_ini) ;

          x_ini  = x_ini + h ;

          y1_ini = y1_ini + (k11+2.0f*k12+2.0f*k13+k14)*(h/6.0f) ;
          y2_ini = y2_ini + (k21+2.0f*k22+2.0f*k23+k24)*(h/6.0f) ;

          H_ini       =   H_th(x_ini, omg_m0, aa1, aa2, cc, y1_ini, y2_ini, omg_k0) ;

     }

tot11bb1x[i] = tot11b1 * h ;



  if ( curv == 0.0f) {
       tot11bb1[i] = tot11bb1x[i];
    }
    else if (curv < 0.0f) {
       tot11bb1[i] = (H0/sqrt(-curv))*sinh((sqrt(-curv)/H0)*tot11bb1x[i]);
    }
    else {
       tot11bb1[i] = (H0/sqrt(curv))*sin((sqrt(curv)/H0)*tot11bb1x[i]);
    }







H_theo_bao1[i] = H_ini ;








//printf("%f\n", H_theo_bao1[i]);

a_th1[i] = (pow(omg_m0,0.50f))*(pow(((pow(tot11bb1[i],2.0f))/((pow((( 1.0f - x1b1[i])/x1b1[i]),2.0f)* H_theo_bao1[i]))),(1.0f/3.0f))) ;


//Dv[i] = pow(((pow(c,3.0f)*z1[i]*pow((tot11bb1[i]),2.0f))/(pow(H_0 ,3.0f)*(H_theo_bao1[i]))),(1.0f/3.0f)) ;


//d_th[i][0] = rs/Dv[i] ;


//del_d1[i][0] = d_th[i][0] - dz1_ob[i][0] ;

       buf1      =   buf1 + pow((( az1_ob[i]  -  a_th1[i] )  / sig_az2[i]),2.0f)	;


//printf("%f\t%f\t%f\t%f\n", z1[i], Dv[i], d_th[i][0], H_theo_bao1[i]);


}


  chi_bao1 = buf1 ;
  
  *y11 = y1_ini ;
  *y22 = y2_ini ;

  
  /*

///////////// Finding the transpose of matrix del_d1

    for (i = 0; i < 3; ++i)
        for (j = 0; j < 1; ++j) {
            trans_del_d1[j][i] = del_d1[i][j];
        }


////////////  finding  matmul(cov_d,del_d1)  

for (i =0; i < 3; i++) {  // rowsfirst

  for (j =0; j < 1; j++) {  //columnsecond

    matmul11[i][j] = 0;
     for (kk=0; kk< 3; kk++) {  //columnfirst
       matmul11[i][j] = matmul11[i][j] + cov_d[i][kk]*del_d1[kk][j];

} 

} 
}


////////// finding matmul(trans_del_d1,matmul11)

for (i =0; i < 1; i++) {  // rowsfirst

  for (j =0; j < 1; j++) {  //columnsecond

    matmul12[i][j] = 0;
     for (kk=0; kk< 3; kk++) {  //columnfirst
       matmul12[i][j] = matmul12[i][j] + trans_del_d1[i][kk]*matmul11[kk][j];

} 

} 
}

// end calculations for dz1


*/

///begin calculations of az2



for ( i = 0; i <= num2-1; i++)  {




x_fin   = x1b2[i] ;
x_ini	= 0.0003f ;
//h       = 0.00050f ;
h       = 0.0003f ;
//tot11b2  = 0.0f ;
y2_ini  = 0.0f ;

//v1      = (1.0f - omg_m0 )*(3.0f/2.0f)*(pow(H_0,2.0f))*(1.0f - w0)* 1.0f/cosh(l0*y2_ini)  ;

y1_ini  = 0.5f ;

H_ini   =   H_th(x_ini, omg_m0, aa1, aa2, cc, y1_ini, y2_ini, omg_k0) ;


p    = (x_fin - x_ini)/h ;



      for ( j = 0; j<= p-1 ; j++)  {

      //    tot11b2 =   tot11b2  + ( H_0/((pow(x_ini,2.0f))*H_ini)) ;
             
          k11 = dy1(x_ini, aa1, aa2, cc, y1_ini, y2_ini, H_ini) ;
          k21 = dy2(x_ini, aa1, aa2, cc, y1_ini, y2_ini, H_ini) ;

          k12 = dy1(x_ini + 0.50f*h, aa1, aa2, cc, y1_ini + k11*(h/2.0f), y2_ini + k21*(h/2.0f), H_ini) ;
          k22 = dy2(x_ini + 0.50f*h, aa1, aa2, cc, y1_ini + k11*(h/2.0f), y2_ini + k21*(h/2.0f), H_ini) ;

          k13 = dy1(x_ini + 0.50f*h, aa1, aa2, cc, y1_ini + k12*(h/2.0f), y2_ini + k22*(h/2.0f), H_ini) ;
          k23 = dy2(x_ini + 0.50f*h, aa1, aa2, cc, y1_ini + k12*(h/2.0f), y2_ini + k22*(h/2.0f), H_ini) ;

          k14 = dy1(x_ini + h, aa1, aa2, cc, y1_ini + k13*h, y2_ini + k23*h, H_ini) ;
          k24 = dy2(x_ini + h, aa1, aa2, cc, y1_ini + k13*h, y2_ini + k23*h, H_ini) ;

          x_ini  = x_ini + h ;

          y1_ini = y1_ini + (k11+2.0f*k12+2.0f*k13+k14)*(h/6.0f) ;
          y2_ini = y2_ini + (k21+2.0f*k22+2.0f*k23+k24)*(h/6.0f) ;

          H_ini       =   H_th(x_ini, omg_m0, aa1, aa2, cc, y1_ini, y2_ini, omg_k0) ;
//printf("%f\n", x_ini);
     }



//x_fin   = x1b2[i] ;
//x_ini	= 0.0000001 ;
//h       = 0.00050f ;
h       = 0.0003f ;
tot11b2  = 0.0f ;
//y2_ini  = 0.0f ;

//v1      = (1.0f - omg_m0 )*(3.0f/2.0f)*(pow(H_0,2.0f))*(1.0f - w0)* 1.0f/cosh(l0*y2_ini)  ;

//y1_ini  = 10.0f;

H_ini   =   H_th(x_ini, omg_m0, aa1, aa2, cc, y1_ini, y2_ini, omg_k0) ;


p    = (1.0f - x_ini)/h ;



      for ( j = 0; j<= p-1 ; j++)  {

          tot11b2 =   tot11b2  + ( 1.0f/((pow(x_ini,2.0f))*H_ini)) ;
             
          k11 = dy1(x_ini, aa1, aa2, cc, y1_ini, y2_ini, H_ini) ;
          k21 = dy2(x_ini, aa1, aa2, cc, y1_ini, y2_ini, H_ini) ;

          k12 = dy1(x_ini + 0.50f*h, aa1, aa2, cc, y1_ini + k11*(h/2.0f), y2_ini + k21*(h/2.0f), H_ini) ;
          k22 = dy2(x_ini + 0.50f*h, aa1, aa2, cc, y1_ini + k11*(h/2.0f), y2_ini + k21*(h/2.0f), H_ini) ;

          k13 = dy1(x_ini + 0.50f*h, aa1, aa2, cc, y1_ini + k12*(h/2.0f), y2_ini + k22*(h/2.0f), H_ini) ;
          k23 = dy2(x_ini + 0.50f*h, aa1, aa2, cc, y1_ini + k12*(h/2.0f), y2_ini + k22*(h/2.0f), H_ini) ;

          k14 = dy1(x_ini + h, aa1, aa2, cc, y1_ini + k13*h, y2_ini + k23*h, H_ini) ;
          k24 = dy2(x_ini + h, aa1, aa2, cc, y1_ini + k13*h, y2_ini + k23*h, H_ini) ;

          x_ini  = x_ini + h ;

          y1_ini = y1_ini + (k11+2.0f*k12+2.0f*k13+k14)*(h/6.0f) ;
          y2_ini = y2_ini + (k21+2.0f*k22+2.0f*k23+k24)*(h/6.0f) ;

          H_ini       =   H_th(x_ini, omg_m0, aa1, aa2, cc, y1_ini, y2_ini, omg_k0) ;
//printf("%f\n", x_ini);
     }





tot11bb2x[i] = tot11b2 * h ;






  if ( curv == 0.0f) {
       tot11bb2[i] = tot11bb2x[i];
    }
    else if (curv < 0.0f) {
       tot11bb2[i] = (H0/sqrt(-curv))*sinh((sqrt(-curv)/H0)*tot11bb2x[i]);
    }
    else {
       tot11bb2[i] = (H0/sqrt(curv))*sin((sqrt(curv)/H0)*tot11bb2x[i]);
    }










H_theo_bao2[i] = H_ini ;

//printf("%f\n", tot11bb2[i]);

a_th2[i][0] = (pow(omg_m0,0.50f))*(pow(((pow(tot11bb2[i],2.0f))/((pow((( 1.0f - x1b2[i])/x1b2[i]),2.0f)* H_theo_bao2[i]))),(1.0f/3.0f))) ;

//Dv2[i] = pow(((pow(c,3.0f)*z2[i]*pow((tot11bb2[i]),2.0f))/(pow(H_0 ,3.0f)*(H_theo_bao2[i]))),(1.0f/3.0f)) ;



//d_th2[i] = rs/Dv2[i] ;



del_A2[i][0] = a_th2[i][0] - az2_ob[i][0] ;



//printf("%f\t%f\t%f\t%f\n", z2[i], Dv2[i], d_th2[i], a_th2[i][0]);


}




///////////// Finding the transpose of matrix del_A2

    for (i = 0; i < 3; ++i)
        for (j = 0; j < 1; ++j) {
            trans_del_A2[j][i] = del_A2[i][j];
        }


////////////  finding  matmul(cov_a,del_A2)  

for (i =0; i < 3; i++) {  // rowsfirst

  for (j =0; j < 1; j++) {  //columnsecond

    matmul21[i][j] = 0;
     for (kk=0; kk< 3; kk++) {  //columnfirst
       matmul21[i][j] = matmul21[i][j] + cov_a[i][kk]*del_A2[kk][j];

} 

} 
}


////////// finding matmul(trans_del_A2,matmul21)

for (i =0; i < 1; i++) {  // rowsfirst

  for (j =0; j < 1; j++) {  //columnsecond

    matmul22[i][j] = 0;
     for (kk=0; kk< 3; kk++) {  //columnfirst
       matmul22[i][j] = matmul22[i][j] + trans_del_A2[i][kk]*matmul21[kk][j];

} 

} 
}


// end calculations for Az2


   matmul_sum[0][0] = matmul22[0][0] ;

   chi_bao = matmul_sum[0][0] + chi_bao1 ;

///////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////E N D  B A O//////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////






///////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////B E G I N  H U B B L E////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////


sum1h = 0.0f ;


for ( i = 0; i <= num1h-1; i++)  {

x_fin   = x1[i] ;
x_ini	= 0.0003f ;
//h       = 0.00050f ;
h       = 0.0003f ;

y2_ini  = 0.0f ;

//v1      = (1.0f - omg_m0 )*(3.0f/2.0f)*(pow(H_0,2.0f))*(1.0f - w0)* 1.0f/cosh(l0*y2_ini)  ;

y1_ini  = 0.5f ;

H_ini   =   H_th(x_ini, omg_m0, aa1, aa2, cc, y1_ini, y2_ini, omg_k0) ;


p    = (x_fin - x_ini)/h ;
//printf("%d\n", p);
//printf("%lf\n", H_ini);
      for ( j = 0; j<= p-1 ; j++)  {
             
          k11 = dy1(x_ini, aa1,aa2, cc, y1_ini, y2_ini, H_ini) ;
          k21 = dy2(x_ini, aa1, aa2, cc, y1_ini, y2_ini, H_ini) ;

          k12 = dy1(x_ini + 0.50f*h, aa1, aa2, cc, y1_ini + k11*(h/2.0f), y2_ini + k21*(h/2.0f), H_ini) ;
          k22 = dy2(x_ini + 0.50f*h, aa1, aa2, cc, y1_ini + k11*(h/2.0f), y2_ini + k21*(h/2.0f), H_ini) ;

          k13 = dy1(x_ini + 0.50f*h, aa1, aa2, cc, y1_ini + k12*(h/2.0f), y2_ini + k22*(h/2.0f), H_ini) ;
          k23 = dy2(x_ini + 0.50f*h, aa1, aa2, cc, y1_ini + k12*(h/2.0f), y2_ini + k22*(h/2.0f), H_ini) ;

          k14 = dy1(x_ini + h, aa1, aa2, cc, y1_ini + k13*h, y2_ini + k23*h, H_ini) ;
          k24 = dy2(x_ini + h, aa1, aa2, cc, y1_ini + k13*h, y2_ini + k23*h, H_ini) ;

          x_ini  = x_ini + h ;

          y1_ini = y1_ini + (k11+2.0f*k12+2.0f*k13+k14)*(h/6.0f) ;
          y2_ini = y2_ini + (k21+2.0f*k22+2.0f*k23+k24)*(h/6.0f) ;

          H_ini       =   H_th(x_ini, omg_m0, aa1, aa2, cc, y1_ini, y2_ini, omg_k0) ;
//printf("%lf\n", x_ini);
     }

//printf("%lf\n", H_ini);
 H_theo[i] = H_ini;

//printf("%lf\n", y1_ini);

sigma[i] = (sig_obh[i]/H0) + ((sig_H0/H0)*H_theo[i]) ;

sum1h = sum1h + (pow((H_theo[i] - (H_ob[i]/H0)), 2.0f))/(pow((sigma[i]), 2.0f)) ;

 //printf("%f\t%f\t%f\n",z1h[i], H_theo[i],  H_ob[i] ) ;


 }

chi_hubble= sum1h ;


///////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////E N D  H U B B L E///////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////


*chi = chi_hubble + chi_bao ;


//printf("%f\t%f\n", chi_hubble, chi_bao);


  // prob = exp(- chi/2.0f);



//printf("%f\n", prob);

//    return chi ;

}


int main(int argc, char **argv)
{
	

     long seed1 ;
     double  cc_max, cc_min, cc_new, omg_m0new, aa2_max, aa2_min, aa2_new, omg_k0new ;
     FILE  * fp4, *fp9;
     int i, j,  accept ;
     double ran, buf1, G ;
     double w0_new,   step_m0, step_aa2, step_w, step_cc, step_k0 ;
     double   omg_m0min,  w0_min, omg_m0max, w0_max, omg_k0min, omg_k0max;
     FILE *fp ,  *fp1, *fp2, *fp3, *fp5, *fp6;
     char    filename[100] ;
   long  double  u_old, u_new, aa1, aa1_min, aa1_new, step_aa1, aa1_max;
   double chi1, chi2 ;
     double omg_m0, aa2, w0, cc, omg_k0 ;     
     double u1, u2;
     double H_0, H0_new, H0_min, H0_max, step_H0;
    
     double  chi_nova,  sum1, sum2, sum3 ;
     double y1old, y1new, y2old, y2new, curv ;
     
     double chol_dc[5][5] ;

      sprintf(filename,"d1_ob_1.txt") ;
      fp = fopen(filename, "r");
      for (i=0; i< num1; i++)
        {
        fscanf(fp, "%lf ", &dz1_ob[i][0] ) ;
        
   //     printf("%f\t", dz1_ob[i][1]);  
        }

      sprintf(filename,"A2_ob.txt") ;
      fp1 = fopen(filename, "r");
      for (i=0; i< num2; i++)
        {
        fscanf(fp1, "%lf ", &az2_ob[i][0] ) ;
        
  //       printf("%f\t", az2_ob[i][1]);   
        }


      sprintf(filename,"cov_inv_d_1.txt") ;
      fp2 = fopen(filename, "r");
      for (i=0;i<num1;i++)
        { 
        //printf("\n");
          for (j=0;j<num1;j++)
           {
           fscanf(fp2,"%lf",&cov_d[i][j]);
     //     printf("%f\t", cov_d[i][j]);
           }
        // printf("%f\n",x1[i][j]);
        }   


      sprintf(filename,"cov_inv_A.txt") ;
      fp3 = fopen(filename, "r");

      for (i=0;i<num2;i++)
        { 
          //printf("\n");
           for (j=0;j<num2;j++)
               {
                 fscanf(fp3,"%lf",&cov_a[i][j]);
             //   printf("%f\t", cov_a[i][j]);
               }
              // printf("%f\n",cov_a[i][j]);
        } 


  double h_z[num], h_mu_ob[num], h_sig_ob11[num], h_sig_ob[num], mu_th[num], tot11ss[num], tot11ssx[num] ;







///     double *h_z = (double*)malloc(num*sizeof(double));
 ///    double *h_mu_ob     = (double*)malloc(num*sizeof(double));
 ///    double *h_sig_ob11        = (double*)malloc(num*sizeof(double));
     //double *h_x1s    = (double*)malloc(num*sizeof(double));
 ///    double *h_sig_ob     = (double*)malloc(num*sizeof(double));
     //double *h_H_theo_n    = (double*)malloc(num*sizeof(double));
///     double *mu_th     = (double*)malloc(num*sizeof(double));
///     double *tot11ss    = (double*)malloc(num*sizeof(double));
     

     //double *h_z, *h_mu_ob, *h_sig_ob11, *h_x1s, *h_sig_ob, *h_H_theo_n, *mu_th, *tot11ss;
     double *h_x1s, *h_H_theo_n;
     //cudaMallocHost((double **)&h_z, num * sizeof(double));
     //cudaMallocHost((double **)&h_mu_ob, num * sizeof(double));
     //cudaMallocHost((double **)&h_sig_ob11, num * sizeof(double));
     cudaMallocHost((double **)&h_x1s, num * sizeof(double));
     //cudaMallocHost((double **)&h_sig_ob, num * sizeof(double));
	
     cudaMallocHost((double **)&h_H_theo_n, num * sizeof(double));
	
     //cudaMallocHost((double **)&mu_th, num * sizeof(double));
     //cudaMallocHost((double **)&tot11ss, num * sizeof(double));



       sprintf(filename,"pan_data.txt") ;
       fp5 = fopen(filename, "r");
       for (i=0; i< num; i++)
        {
        fscanf(fp5, "%le %le %le ", &h_z[i],&h_mu_ob[i], &h_sig_ob11[i]) ;
        h_x1s[i] = 1.0f/(1.0f + h_z[i]) ;
        h_sig_ob[i] = 1.0f * h_sig_ob11[i] ;
      //  printf("%le\t%le\t%le\t%le\n", h_z[i], h_mu_ob[i],h_sig_ob11[i],h_x1s[i]);  
        }
        
       sprintf(filename,"H_data.txt") ;
       fp6 = fopen(filename, "r");
       for (i=0; i< num1h; i++)
        {
        fscanf(fp6, "%lf %lf %lf ", &z1h[i],&H_ob[i], &sig_obh[i]) ;
        x1[i] = 1.0f/(1.0f + z1h[i]);
       
      //  printf("%f\t%f\t%f\t%f\n", h_z[i], h_mu_ob[i],h_sig_ob11[i],h_x1s[i]);  
        }


     /*
      
      / sprintf(filename,"redshift_new.txt");
      fp6=fopen(filename,"r")	;
      for (i = 0; i < num1h; i++)
      { 
         fscanf(fp6, "%f", &z1h[i]) ;
	x1[i] = 1.0f/(1.0f + z1h[i]);
      }
      sprintf(filename,"Obs_H_new.txt")	;
      fp7=fopen(filename,"r");
      for (i = 0; i < num1h; i++)
      { 
         fscanf(fp7, "%f", &H_ob[i]);
      }
      sprintf(filename,"sigma_z_new.txt");
      fp8=fopen(filename,"r")	;
      for (i = 0; i < num1h; i++)
      { 
         fscanf(fp8, "%f", &sig_obh[i]);
      }

*/
    
    double alpha_norm = 0.6f ;



      sprintf(filename,"c333_thinned_498_covm_CHDC.txt") ;
      fp9 = fopen(filename, "r");

      for (i=0;i<5;i++)
        { 
        //printf("\n");
          for (j=0;j<5;j++)
           {
           fscanf(fp9,"%le",&chol_dc[i][j]);
        // printf("%lf\t", chol_dc[i][j]);
           }
       // printf("%lf\t", chol_dc[i][j]);
        }
    
    
	double  *d_x1s, *d_H_theo_n;

	cudaMalloc((double **)&d_x1s, num * sizeof(double));
	cudaMalloc((double **)&d_H_theo_n, num * sizeof(double));

 	cudaMemcpy(d_x1s, h_x1s, num * sizeof(double), cudaMemcpyHostToDevice);
 	cudaMemcpy(d_H_theo_n, h_H_theo_n, num * sizeof(double), cudaMemcpyHostToDevice);


       double *omg_m0_arr = (double*)malloc(nsteps1*sizeof(double));
       double *l0_arr     = (double*)malloc(nsteps1*sizeof(double));
       double *val        = (double*)malloc(nsteps1*sizeof(double));
       double *v1_arr     = (double*)malloc(nsteps1*sizeof(double));
       double *w0_arr     = (double*)malloc(nsteps1*sizeof(double));
       double *chi_arr    = (double*)malloc(nsteps1*sizeof(double));
       double *H0_arr     = (double*)malloc(nsteps1*sizeof(double));

cc_max      = 5.0f ;
aa1_max     = 0.01L;///4.0f ;
aa2_max     = 0.035f;
omg_m0max   = 1.0f ;  
omg_k0max   = 0.2f ;      
//w0_max     = 0.0f ;
//H0_max     = 80.0f ;

cc_min     = 0.0f ;
aa1_min     = 0.0L ;
aa2_min     = 0.0f ;
omg_m0min  = 0.0f ;
omg_k0min  = -0.2f ;
//w0_min     = - 1.0f ;
//H0_min	   = 60.0f ;

//step_m0    = 0.01f ;
//step_aa1    = 0.0000006L;//0.03f;//////////0.1f ;
//step_aa2    = 0.005f;
//step_w    = 0.05f;
//step_cc    = 0.08f ;
//step_k0	   = 0.009f ;
//step_H0	  = 0.1f ;


omg_m0    = 0.269f ;
aa1       = 0.000001L;         ///0.2f ; 
aa2	  = 0.02f ; 
//w0        = - 0.98f ; //- 0.90f ; 
cc        = 2.3f ;
omg_k0    = -0.0421f ;
//H_0	  = 68.0f ;


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	supernova<< <32 , 64 >> > ( d_x1s, num, omg_m0, aa1, aa2, cc, d_H_theo_n, omg_k0 );
	cudaDeviceSynchronize();	
 	
	cudaMemcpy(h_H_theo_n, d_H_theo_n,  num * sizeof(double), cudaMemcpyDeviceToHost);



 	 curv = -(omg_k0)*pow(H0 ,2.0f) ;


	sum1 = 0.0f ;
	sum2 = 0.0f ;
	sum3 = 0.0f ;


	for (int ii = 0; ii <= num-1; ii++)  {

	tot11ssx[ii] = h_H_theo_n[ii] ;
	
	


  if ( curv == 0.0f) {
       tot11ss[ii] = tot11ssx[ii];
    }
    else if (curv < 0.0f) {
       tot11ss[ii] = (H0/sqrt(-curv))*sinh((sqrt(-curv)/H0)*tot11ssx[ii]);
    }
    else {
       tot11ss[ii] = (H0/sqrt(curv))*sin((sqrt(curv)/H0)*tot11ssx[ii]);
    }




	
	mu_th[ii] = 5.0f*log10(3000.0f*tot11ss[ii]*1.0f/h_x1s[ii])+ 25.0f-(5.0f*log10(H0/100.0f)) ; //!!!!theoretical mu

//printf("%f\t%f\t%f\n", h_z[ii], h_H_theo_n[ii], mu_th[ii]);

	}




	for (int ii = 0; ii <= num-1 ; ii ++) {

	sum1 = sum1 + ((pow((h_mu_ob[ii] - mu_th[ii]),2.0f))/(pow(h_sig_ob[ii],2.0f))) ;

	sum2 = sum2 + ((h_mu_ob[ii] - mu_th[ii])/(pow(h_sig_ob[ii],2.0f))) ;

	sum3 = sum3 + (1.0f/(pow(h_sig_ob[ii],2.0f))) ;
	
	}

	chi_nova = sum1-((pow(sum2 ,2.0f))/sum3);//- (2.0f * log(10.0f) * sum2)/(5.0f * sum3) ;


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

find_l(omg_m0, aa1, aa2, cc, &y1old, &y2old, &u1, omg_k0) ;
chi1 = chi_nova + u1 ;

 u_old = expl(- chi1/2.0);
//printf("%lf\n", chi_nova);


    
    
    

    j = 0 ;

    for ( i =0; i<nsteps; i++) {

      generate1 : G = gasdev(&seed1) ;
       //omg_m0new = omg_m0 + step_m0 * G ;
        omg_m0new = omg_m0 + alpha_norm*( (chol_dc[0][0]*G) + (chol_dc[0][1]*G) + (chol_dc[0][2]*G) + (chol_dc[0][3]*G ) + (chol_dc[0][4]*G ) )    ;  
       if (omg_m0new <= omg_m0min || omg_m0new >= omg_m0max) 
       goto generate1 ;
       
       generate2 : G = gasdev(&seed1) ;
       omg_k0new = omg_k0 +  alpha_norm*( (chol_dc[1][0]*G) + (chol_dc[1][1]*G) + (chol_dc[1][2]*G) + (chol_dc[1][3]*G ) + (chol_dc[1][4]*G ))    ; 
       if (omg_k0new <= omg_k0min || omg_k0new >= omg_k0max)  
       goto generate2 ;
       
       
//printf("%lf\n", G);
       generate3 : G = gasdev(&seed1) ;
            aa1_new = aa1 +  alpha_norm*( (chol_dc[2][0]*G) + (chol_dc[2][1]*G) + (chol_dc[2][2]*G) + (chol_dc[2][3]*G ) + (chol_dc[2][4]*G ))    ; 
        if (aa1_new <= aa1_min || aa1_new >= aa1_max) 
       goto generate3 ;
       
//printf("%Le\n", aa1_new);
       generate4 : G = gasdev(&seed1) ;
           aa2_new = aa2 +  alpha_norm*( (chol_dc[3][0]*G) + (chol_dc[3][1]*G) + (chol_dc[3][2]*G) + (chol_dc[3][3]*G ) + (chol_dc[3][4]*G ))    ; 
       if (aa2_new <= aa2_min || aa2_new >= aa2_max) 
       goto generate4 ;
       
      // if ( aa1_new > aa2_new)
      // goto generate2 ;
       
       
//printf("%lf\n", G);
       generate5 : G = gasdev(&seed1) ;
            cc_new = cc +  alpha_norm*( (chol_dc[4][0]*G) + (chol_dc[4][1]*G) + (chol_dc[4][2]*G) + (chol_dc[4][3]*G ) + (chol_dc[4][4]*G ))    ; 
       if (cc_new <= cc_min || cc_new >= cc_max )
       goto generate5 ;




/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	supernova<< <32 , 64 >> > ( d_x1s, num, omg_m0new, aa1_new, aa2_new, cc_new, d_H_theo_n, omg_k0new );
	cudaDeviceSynchronize();	
 	
 	cudaMemcpy(h_H_theo_n, d_H_theo_n,  num * sizeof(double), cudaMemcpyDeviceToHost);
 	
 	
 	 curv = -(omg_k0new)*pow(H0 ,2.0f) ;
 	

        sum1 = 0.0f ;
	sum2 = 0.0f ;
	sum3 = 0.0f ;

	for ( int ii = 0; ii <= num-1; ii++)  {
		
	
	tot11ssx[ii] = h_H_theo_n[ii] ;
	
	
	


  if ( curv == 0.0f) {
       tot11ss[ii] = tot11ssx[ii];
    }
    else if (curv < 0.0f) {
       tot11ss[ii] = (H0/sqrt(-curv))*sinh((sqrt(-curv)/H0)*tot11ssx[ii]);
    }
    else {
       tot11ss[ii] = (H0/sqrt(curv))*sin((sqrt(curv)/H0)*tot11ssx[ii]);
    }
	
	
	

	mu_th[ii] = 5.0f*log10(3000.0f*tot11ss[ii]*1.0f/h_x1s[ii])+ 25.0f-(5.0f*log10(H0/100.0f)) ; //!!!!theoretical mu

	//printf("%f\t%f\t%f\n", z[i], mu_th[i], mu_ob[i]);

	}




	for (int ii = 0; ii <= num-1 ; ii ++) {

	sum1 = sum1 + ((pow((h_mu_ob[ii] - mu_th[ii]),2.0f))/(pow(h_sig_ob[ii],2.0f))) ;

	sum2 = sum2 + ((h_mu_ob[ii] - mu_th[ii])/(pow(h_sig_ob[ii],2.0f))) ;

	sum3 = sum3 + (1.0f/(pow(h_sig_ob[ii],2.0f))) ;

	}


	chi_nova = sum1-((pow(sum2 ,2.0f))/sum3);//- (2.0f * log(10.0f) * sum2)/(5.0f * sum3) ;


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

find_l(omg_m0new, aa1_new, aa2_new, cc_new, &y1new, &y2new, &u2, omg_k0new) ;
chi2 = chi_nova + u2 ;


 u_new = expl(- chi2/2.0f);
//printf("%lf\n", chi2);



       accept =0;

      if (u_new > u_old)  {
         accept =1;
      }
      if (u_new < u_old) {
        ran = ran1(&seed1);
        buf1 = u_new/u_old;

      if (buf1 > ran) {
        accept =1;
        }
        }
      //printf("%d\t%f\t%f\t%f\t%f\n",jj, ran, buf1, u_old, u_new);
      if (accept == 1) {

      omg_m0    = omg_m0new  ;
      aa1       = aa1_new ;
      aa2	= aa2_new ;
      cc        = cc_new ;
      omg_k0	= omg_k0new ;
      
     // w0        = w0_new ;
      u_old     = u_new  ;
     // H_0       = H_0  ;
      y1old	= y1new ;
      y2old	= y2new ;

      val[j]          = u_new ;
      chi_arr[j]      = -2.0f*log(u_new) ;

      omg_m0_arr[j]   = omg_m0 ;
      l0_arr[j]       = aa1 ;
      v1_arr[j]       = cc ;
      
                        
      
      w0_arr[j]       = aa2 ;//((y2old*y2old)/2.0f - ((cc)*(1.0f + exp((-y1old)/pow((6.0f*aa), 0.5f)))))/ ((y2old*y2old)/2.0f + ((cc)*(1.0f + exp((-y1old)/pow((6.0f*aa), 0.5f))))) ;
      H0_arr[j]	      = omg_k0 ;
      
     
  //    ((y2old*y2old)/2.0f - ((aa*cc*cc)*cosh(y1old/(pow((6.0f*aa), 0.5f)))))/ ((y2old*y2old)/2.0f + ((aa*cc*cc)*cosh(y1old/(pow((6.0f*aa), 0.5f)))))
      
    

      //fprintf(fp4, "%f\t%f\t%f\n",omg_m0_arr[jj], omg_l_arr[jj], u_arr[jj] );
      //printf("%d\t%f\n", jj, ran);

      j  =  j +1 ;	
      }    
    printf("%d\t%d\n", i, j) ;		
  // i = i + 1 ;						
   }



  double ratio = (double) j/nsteps;
  printf("Acceptence Ratio = %f\n", ratio); 
  fp4 = fopen("/home/image/Francy/de_obs/margarita_new/joint/final_nonflat/results/c333_0.666.txt" ,"w");
    for ( i = 0;i < j-1; i++)
       {
          fprintf(fp4, "%lf\t%lf\t%le\t%lf\t%lf\t%lf\t%le\n",omg_m0_arr[i], H0_arr[i], l0_arr[i],w0_arr[i], v1_arr[i], chi_arr[i],  val[i] ); 
       }

 // fp11 = fopen("/home/image/Francy/de_obs/l_model/results/first_run/22.txt" ,"w");
 //   for ( i = 0;i < j-1; i++)
 //      {
 //         fprintf(fp11, "%f\t%f\t%f\t%f\n",omg_m0_arr[i], l0_arr[i], phi_arr[i], w0_arr[i] ); 
 //      }
  fclose(fp4)		;

  free(omg_m0_arr);
  free(l0_arr);
  free(v1_arr);
  free(w0_arr);
  free(chi_arr);
  free(val);
  free(H0_arr) ;
							

//    fclose(fp11)		;
  
 ///	free(tot11ss);
///	free(mu_th);
  ///	free(h_z);
///	free(h_mu_ob);
///	free(h_sig_ob11);
//	cudaFreeHost(h_x1s);
///	free(h_sig_ob);
//	cudaFreeHost(h_H_theo_n);

//	cudaFree(d_x1s);
//	cudaFree(d_H_theo_n);
	
//cudaDeviceReset();
  
  
  
  return EXIT_SUCCESS;								
  }
























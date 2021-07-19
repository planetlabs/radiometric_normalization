#include <assert.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#include "smapa.h"

#include "fail.c"
#include "xmalloc.c"
#include "blur.c"

typedef float (*interpolation_operator_float)(
		float *image, int width, int height,
		float x, float y);

typedef float (*extension_operator_float)(
		float *image, int width, int height,
		int i, int j);

static float extend_float_image_constant(float *xx, int w, int h, int i, int j)
{
	float (*x)[w] = (void*)xx;
	if (i < 0) i = 0;
	if (j < 0) j = 0;
	if (i >= w) i = w - 1;
	if (j >= h) j = h - 1;
	return x[j][i];
}

static float cell_interpolate_bilinear(float a, float b, float c, float d,
					float x, float y)
{
	float r = 0;
	r += a*(1-x)*(1-y);
	r += b*(1-x)*(y);
	r += c*(x)*(1-y);
	r += d*(x)*(y);
	return r;
}

static float cell_interpolate_nearest(float a, float b, float c, float d,
					float x, float y)
{
	// return a;
	if (x<0.5) return y<0.5 ? a : b;
	else return y<0.5 ? c : d;
}

static float cell_interpolate(float a, float b, float c, float d,
					float x, float y, int method)
{
	switch(method) {
	case 0: return cell_interpolate_nearest(a, b, c, d, x, y);
	//case 1: return marchi(a, b, c, d, x, y);
	case 2: return cell_interpolate_bilinear(a, b, c, d, x, y);
	default: return 0;
	}
}

static float interpolate_float_image_bilinearly(float *x, int w, int h,
		float i, float j)
{
	int ii = i;
	int jj = j;
	extension_operator_float p = extend_float_image_constant;
	float a = p(x, w, h, ii  , jj  );
	float b = p(x, w, h, ii  , jj+1);
	float c = p(x, w, h, ii+1, jj  );
	float d = p(x, w, h, ii+1, jj+1);
	return cell_interpolate(a, b, c, d, i-ii, j-jj, 2);
}

SMART_PARAMETER_SILENT(MAGIC_SIGMA,1.6)
SMART_PARAMETER_SILENT(PRESMOOTH,0)

static void downsa_v2(float *out, float *in,
		int outw, int outh, int inw, int inh)
{
	//fprintf(stderr, "\n\ndownsa_v2 (%d x %d) => (%d x %d)\n\n",
	//		inw,inh,outw,outh);
	if (2*outw != inw) fail("bad horizontal halving");
	if (2*outh != inh) fail("bad vertical halving");

	float (*y)[outw] = (void*)out;
	float (*x)[inw] = (void*)in;

	for (int j = 0; j < outh; j++)
	for (int i = 0; i < outw; i++) {
		float g = 0;
		g += x[2*j+0][2*i+0];
		g += x[2*j+0][2*i+1];
		g += x[2*j+1][2*i+0];
		g += x[2*j+1][2*i+1];
		y[j][i] = g/4;
	}
}

void downscale_image(float *out, float *in,
		int outw, int outh, int inw, int inh,
		float scalestep)
{
	if (scalestep == -2) {downsa_v2(out,in,outw,outh,inw,inh); return;}
	//fprintf(stderr, "downscale(%g): %dx%d => %dx%d\n",
	//		scalestep, inw, inh, outw, outh);

	assert(scalestep > 1);
	assert(scalestep * outw >= inw);
	//assert(scalestep * outw <= inw + 1);
	assert(scalestep * outh >= inh);
	//assert(scalestep * outh <= inh + 1);

	float factorx = inw/(float)outw;
	float factory = inh/(float)outh;

	float blur_size = MAGIC_SIGMA()*sqrt((factorx*factory-1)/3);

	/*fprintf(stderr, "blur_size = %g\n", blur_size);*/

	float *gin = xmalloc(inw * inh * sizeof(float));
	if (outw < inw || outh < inh) {
		void gblur_gray(float*, float*, int, int, float);
		gblur_gray(gin, in, inw, inh, blur_size);
	} else {
		assert(inw == outw);
		assert(inh == outh);
		for (int i = 0; i < inw*inh; i++)
			gin[i] = in[i];
	}

	// XXX ERROR FIXME
	// TODO: zoom by fourier, or zoom by bicubic interpolation
	interpolation_operator_float ev = interpolate_float_image_bilinearly;

	for (int j = 0; j < outh; j++)
	for (int i = 0; i < outw; i++)
	{
		float x = factorx*i;
		float y = factory*j;
		out[outw*j + i] = ev(gin, inw, inh, x, y);
	}

	xfree(gin);
}


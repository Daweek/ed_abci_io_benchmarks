#version 430

#define NTHREADS 256

layout (local_size_x = NTHREADS) in;
// out format 
layout(std430, binding=0) buffer in_vec
{
    vec2 array[];
};

layout(std430, binding=2) buffer in_bor
{
    vec4 d_bor;
};

layout(binding=3) uniform all_uniforms
{
    vec2 rnd;
    int mappings;
    float rot_l;
    float rot_r;
};
// uniform float time;
uniform int n;


shared float cacheMaxX [NTHREADS];
shared float cacheMaxY [NTHREADS];
shared float cacheMinX [NTHREADS];
shared float cacheMinY [NTHREADS];

void main() {
    // Test to compute shader
    // blockIdx.x * blockDim.x + threadIdx.x;
    uint index = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    uint stride = gl_WorkGroupSize.x * gl_NumWorkGroups.x;
    uint offset = 0;

	float maxX = array[index + 0].x;
    float maxY = array[index + 0].y;
    float minX = array[index + 0].x;
    float minY = array[index + 0].y;


    while(index + offset < n){
 		maxX = max(maxX, array[index + offset].x);
 		maxY = max(maxY, array[index + offset].y);

 		minX = min(minX, array[index + offset].x);
 		minY = min(minY, array[index + offset].y);

 		offset += stride;
 	}
    
    cacheMaxX[gl_LocalInvocationID.x] = maxX;
 	cacheMaxY[gl_LocalInvocationID.x] = maxY;
 	cacheMinX[gl_LocalInvocationID.x] = minX;
 	cacheMinY[gl_LocalInvocationID.x] = minY;

  // memoryBarrierShared();
  	barrier();

    // reduction
 	uint i = gl_WorkGroupSize.x/2;
 	while(i != 0){
 		if(gl_LocalInvocationID.x < i){
 			cacheMaxX[gl_LocalInvocationID.x] = max(cacheMaxX[gl_LocalInvocationID.x], cacheMaxX[gl_LocalInvocationID.x + i]);
 			cacheMaxY[gl_LocalInvocationID.x] = max(cacheMaxY[gl_LocalInvocationID.x], cacheMaxY[gl_LocalInvocationID.x + i]);
 			cacheMinX[gl_LocalInvocationID.x] = min(cacheMinX[gl_LocalInvocationID.x], cacheMinX[gl_LocalInvocationID.x + i]);
 			cacheMinY[gl_LocalInvocationID.x] = min(cacheMinY[gl_LocalInvocationID.x], cacheMinY[gl_LocalInvocationID.x + i]);
			// memoryBarrierShared();
			barrier();
 		}

 		//memoryBarrierShared();
    	barrier();
 		i /= 2;
 	}

   
    if(gl_LocalInvocationID.x == 0){
 		
    	d_bor.x = cacheMaxX[0];
 		d_bor.y = cacheMaxY[0];
 		d_bor.z = cacheMinX[0];
 		d_bor.w = cacheMinY[0];
 		// barrier();
 	}
    
}

    
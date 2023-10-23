#version 430

in vec3 colorf;
out vec4 color;

//in int rtype;
uniform int rtype;
// uniform vec2 rnd;
layout(binding=3) uniform all_uniforms
{
    vec2 rnd;
    int mappings;
    float rot_l;
    float rot_r;
};

float rand(vec2 co)
{
    float a = 12.9898*(rnd.x*10);
    float b = 78.233*(rnd.y*10);
    float c = 43758.5453;
    float dt= dot(co.xy ,vec2(a,b));
    float sn= mod(dt,3.14);
    return fract(sin(sn) * c);
}

void main() {

  vec4 gray = vec4(colorf,1.0);
  vec4 black = vec4(0.0,0.0,0.0,1.0);

  
  if (rtype == 1){
      if(rand(gl_FragCoord.xy)<0.5) 
          color = gray;
          // color = vec4(colorf, 1.0);
      else 
          color = black;
  } 
  else{
      color = vec4(colorf, 1.0);
  } 
}
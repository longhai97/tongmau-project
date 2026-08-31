// TongMau.ColorSpace — sRGB <-> CIE L*a*b* (D65) <-> LCh conversions, plus
// circular mean/std for hue (a degrees quantity that wraps at 360).
(function(global){

  function srgbToLinear(c){ c/=255; return c<=0.04045 ? c/12.92 : Math.pow((c+0.055)/1.055,2.4); }
  function linearToSrgb(c){ return c<=0.0031308 ? c*12.92*255 : (1.055*Math.pow(c,1/2.4)-0.055)*255; }
  function clamp255(v){ return v<0?0:(v>255?255:v); }

  function rgbToLab(r,g,b){
    const rl=srgbToLinear(r), gl=srgbToLinear(g), bl=srgbToLinear(b);
    let X = rl*0.4124564+gl*0.3575761+bl*0.1804375;
    let Y = rl*0.2126729+gl*0.7151522+bl*0.0721750;
    let Z = rl*0.0193339+gl*0.1191920+bl*0.9503041;
    X/=0.95047; Z/=1.08883;
    const fx = X>0.008856 ? Math.cbrt(X) : (7.787*X+16/116);
    const fy = Y>0.008856 ? Math.cbrt(Y) : (7.787*Y+16/116);
    const fz = Z>0.008856 ? Math.cbrt(Z) : (7.787*Z+16/116);
    return [116*fy-16, 500*(fx-fy), 200*(fy-fz)];
  }
  function labToRgb(L,a,b){
    const fy=(L+16)/116, fx=a/500+fy, fz=fy-b/200;
    const fx3=fx*fx*fx, fy3=fy*fy*fy, fz3=fz*fz*fz;
    const X=0.95047*(fx3>0.008856?fx3:(fx-16/116)/7.787);
    const Y=1.0*(fy3>0.008856?fy3:(fy-16/116)/7.787);
    const Z=1.08883*(fz3>0.008856?fz3:(fz-16/116)/7.787);
    const rl = X*3.2404542+Y*-1.5371385+Z*-0.4985314;
    const gl = X*-0.9692660+Y*1.8760108+Z*0.0415560;
    const bl = X*0.0556434+Y*-0.2040259+Z*1.0572252;
    return [clamp255(linearToSrgb(rl)), clamp255(linearToSrgb(gl)), clamp255(linearToSrgb(bl))];
  }

  // Polar form of the a/b plane: Chroma (colorfulness) + Hue (degrees, 0-360).
  function labToLch(L,a,b){
    const C = Math.sqrt(a*a+b*b);
    let h = Math.atan2(b,a) * 180/Math.PI;
    if(h<0) h += 360;
    return [L,C,h];
  }
  function lchToLab(L,C,h){
    const rad = h*Math.PI/180;
    return [L, C*Math.cos(rad), C*Math.sin(rad)];
  }

  function imageDataToLab(imageData){
    const px = imageData.data;
    const n = imageData.width*imageData.height;
    const lab = new Float32Array(n*3);
    for(let i=0, p=0; i<n; i++, p+=4){
      const [L,a,b] = rgbToLab(px[p], px[p+1], px[p+2]);
      lab[i*3]=L; lab[i*3+1]=a; lab[i*3+2]=b;
    }
    return lab;
  }

  function wrapDeg180(d){ return ((d+180) % 360 + 360) % 360 - 180; }
  function wrapDeg360(d){ return ((d % 360) + 360) % 360; }

  // Mean/std of an angular quantity (degrees), optionally weighted.
  // Averages unit vectors instead of raw degrees so 359° and 1° land near 0°,
  // not near 180°. Std uses the Mardia circular-std approximation from the
  // mean resultant length R.
  function circularMeanStd(huesDeg, weights){
    const n = huesDeg.length;
    let sx=0, sy=0, wsum=0;
    for(let i=0;i<n;i++){
      const w = weights ? weights[i] : 1;
      if(w<=0) continue;
      const rad = huesDeg[i]*Math.PI/180;
      sx += Math.cos(rad)*w; sy += Math.sin(rad)*w; wsum += w;
    }
    if(wsum<=0) return {mean:0, std:1e-6};
    const mx = sx/wsum, my = sy/wsum;
    let meanDeg = Math.atan2(my,mx)*180/Math.PI;
    if(meanDeg<0) meanDeg += 360;
    const R = Math.min(1, Math.max(1e-6, Math.sqrt(mx*mx+my*my)));
    const stdDeg = Math.sqrt(Math.max(0, -2*Math.log(R))) * 180/Math.PI;
    return {mean: meanDeg, std: Math.max(stdDeg, 1e-6)};
  }

  global.TongMau = global.TongMau || {};
  global.TongMau.ColorSpace = {
    srgbToLinear, linearToSrgb, clamp255,
    rgbToLab, labToRgb, labToLch, lchToLab,
    imageDataToLab, circularMeanStd, wrapDeg180, wrapDeg360
  };
})(window);

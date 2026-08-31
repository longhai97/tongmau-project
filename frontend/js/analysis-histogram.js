// TongMau.Histogram — weighted 1D histogram/CDF construction and matching,
// used by the "Khớp theo histogram/CDF" analysis option to match a full
// tonal distribution instead of just mean/std (Reinhard-style rescale).
// Supports a circular variant for hue (degrees, wraps at 360): the circle is
// rotated so its least-populated bin becomes the cut point, avoiding
// wraparound artifacts at an arbitrary 0/360 boundary.
(function(global){

  function clamp(v,lo,hi){ return v<lo?lo:(v>hi?hi:v); }

  function buildLinear(values, weights, min, max, bins){
    const hist = new Float64Array(bins);
    const n = values.length;
    for(let i=0;i<n;i++){
      const w = weights ? weights[i] : 1;
      if(w<=0) continue;
      const t = clamp((values[i]-min)/(max-min), 0, 0.999999);
      hist[Math.floor(t*bins)] += w;
    }
    const cdf = new Float64Array(bins+1);
    let acc=0;
    for(let i=0;i<bins;i++){ acc+=hist[i]; cdf[i+1]=acc; }
    const total = cdf[bins] || 1e-9;
    for(let i=0;i<=bins;i++) cdf[i]/=total;
    return {circular:false, min, max, bins, cdf};
  }

  function percentileOfLinear(stat, x){
    const t = clamp((x-stat.min)/(stat.max-stat.min), 0, 1);
    const pos = t*stat.bins;
    const i0 = Math.floor(pos), i1 = Math.min(stat.bins, i0+1);
    const frac = pos-i0;
    return stat.cdf[i0]*(1-frac) + stat.cdf[i1]*frac;
  }

  function valueAtPercentileLinear(stat, p){
    p = clamp(p,0,1);
    let lo=0, hi=stat.bins;
    while(lo<hi){ const mid=(lo+hi)>>1; if(stat.cdf[mid]<p) lo=mid+1; else hi=mid; }
    const i1 = clamp(lo,1,stat.bins);
    const i0 = i1-1;
    const c0=stat.cdf[i0], c1=stat.cdf[i1];
    const frac = c1>c0 ? (p-c0)/(c1-c0) : 0;
    const tVal = (i0+frac)/stat.bins;
    return stat.min + tVal*(stat.max-stat.min);
  }

  function buildCircular(valuesDeg, weights, bins){
    const binOf = (deg)=> clamp(Math.floor((((deg%360)+360)%360)/360*bins), 0, bins-1);
    const rough = new Float64Array(bins);
    const n = valuesDeg.length;
    for(let i=0;i<n;i++){
      const w = weights ? weights[i] : 1;
      if(w<=0) continue;
      rough[binOf(valuesDeg[i])] += w;
    }
    let cutBin=0, cutVal=Infinity;
    for(let i=0;i<bins;i++){ if(rough[i]<cutVal){ cutVal=rough[i]; cutBin=i; } }
    const offsetDeg = (cutBin/bins)*360;

    const hist = new Float64Array(bins);
    for(let i=0;i<n;i++){
      const w = weights ? weights[i] : 1;
      if(w<=0) continue;
      const rotated = (((valuesDeg[i]-offsetDeg)%360)+360)%360;
      hist[binOf(rotated)] += w;
    }
    const cdf = new Float64Array(bins+1);
    let acc=0;
    for(let i=0;i<bins;i++){ acc+=hist[i]; cdf[i+1]=acc; }
    const total = cdf[bins] || 1e-9;
    for(let i=0;i<=bins;i++) cdf[i]/=total;
    return {circular:true, offsetDeg, bins, cdf};
  }

  function percentileOfCircular(stat, deg){
    const rotated = (((deg-stat.offsetDeg)%360)+360)%360;
    const t = rotated/360;
    const pos = t*stat.bins;
    const i0 = Math.floor(pos), i1 = Math.min(stat.bins, i0+1);
    const frac = pos-i0;
    return stat.cdf[i0]*(1-frac) + stat.cdf[i1]*frac;
  }

  function valueAtPercentileCircular(stat, p){
    p = clamp(p,0,1);
    let lo=0, hi=stat.bins;
    while(lo<hi){ const mid=(lo+hi)>>1; if(stat.cdf[mid]<p) lo=mid+1; else hi=mid; }
    const i1 = clamp(lo,1,stat.bins);
    const i0 = i1-1;
    const c0=stat.cdf[i0], c1=stat.cdf[i1];
    const frac = c1>c0 ? (p-c0)/(c1-c0) : 0;
    const tVal = (i0+frac)/stat.bins;
    return (((tVal*360)+stat.offsetDeg)%360+360)%360;
  }

  function build(values, opts){
    opts = opts || {};
    const bins = opts.bins || 64;
    if(opts.circular) return buildCircular(values, opts.weights, bins);
    return buildLinear(values, opts.weights, opts.min, opts.max, bins);
  }
  function percentileOf(stat, x){ return stat.circular ? percentileOfCircular(stat,x) : percentileOfLinear(stat,x); }
  function valueAtPercentile(stat, p){ return stat.circular ? valueAtPercentileCircular(stat,p) : valueAtPercentileLinear(stat,p); }

  global.TongMau = global.TongMau || {};
  global.TongMau.Histogram = { build, percentileOf, valueAtPercentile };
})(window);

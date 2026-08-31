// TongMau.Zones — soft tonal-zone membership (Shadows/Mid/Highlights) used
// by the "Phân vùng theo dải sáng" analysis option. Each pixel gets a
// membership weight per zone based on its own L (0-100); weights are Gaussian
// bumps normalized to sum to 1, so zones blend smoothly instead of producing
// hard cutoffs (which would band the result at zone boundaries).
(function(global){

  const ZONES = [
    {key:'shadows',    label:'Shadows',    center:15, spread:25},
    {key:'mid',        label:'Mid',        center:50, spread:25},
    {key:'highlights', label:'Highlights', center:85, spread:25},
  ];

  function gaussian(x, center, spread){
    const d = (x-center)/spread;
    return Math.exp(-d*d);
  }

  // Returns {shadows:Float32Array, mid:Float32Array, highlights:Float32Array},
  // one entry per pixel, weights normalized to sum to 1 across zones.
  function computeZoneWeights(Lvalues){
    const n = Lvalues.length;
    const weights = {};
    ZONES.forEach(z=>{ weights[z.key] = new Float32Array(n); });
    const raw = new Float32Array(ZONES.length);
    for(let i=0;i<n;i++){
      let sum=0;
      for(let z=0; z<ZONES.length; z++){
        const w = gaussian(Lvalues[i], ZONES[z].center, ZONES[z].spread);
        raw[z] = w; sum += w;
      }
      for(let z=0; z<ZONES.length; z++){
        weights[ZONES[z].key][i] = sum>0 ? raw[z]/sum : 1/ZONES.length;
      }
    }
    return weights;
  }

  global.TongMau = global.TongMau || {};
  global.TongMau.Zones = { ZONES, computeZoneWeights };
})(window);

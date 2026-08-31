// TongMau.StyleMatch — combines color-space.js, analysis-zones.js and
// analysis-histogram.js into the actual reference->target color matching
// used by the "Nhanh (LAB)" mode. Three independent, combinable options:
//   - zone:      match a/b (or C/h) per tonal zone (Shadows/Mid/Highlights)
//                instead of one global statistic, for split-toning-aware looks.
//   - lch:       match in polar Chroma/Hue instead of Cartesian a/b.
//   - histogram: match the full distribution (CDF) instead of just mean/std.
// L (lightness) is always a single global channel (zoning is based on L
// itself), but still respects the histogram option.
(function(global){

  const CHANNEL_DOMAIN = {
    L: {min:0, max:100},
    a: {min:-120, max:120},
    b: {min:-120, max:120},
    C: {min:0, max:150},
  };

  function extractChannels(lab, useLch){
    const CS = global.TongMau.ColorSpace;
    const n = lab.length/3;
    const Lv = new Float32Array(n);
    const c1 = new Float32Array(n); // a, or C if useLch
    const c2 = new Float32Array(n); // b, or h (degrees) if useLch
    for(let i=0;i<n;i++){
      const L=lab[i*3], a=lab[i*3+1], b=lab[i*3+2];
      Lv[i]=L;
      if(useLch){
        const lch = CS.labToLch(L,a,b);
        c1[i]=lch[1]; c2[i]=lch[2];
      } else {
        c1[i]=a; c2[i]=b;
      }
    }
    return {Lv, c1, c2};
  }

  // weighted mean/std (weights null => every sample counts once)
  function weightedMoment(values, weights){
    const n = values.length;
    let sw=0, sx=0;
    for(let i=0;i<n;i++){ const w=weights?weights[i]:1; sw+=w; sx+=w*values[i]; }
    const mean = sw>0 ? sx/sw : 0;
    let svar=0;
    for(let i=0;i<n;i++){ const w=weights?weights[i]:1; const d=values[i]-mean; svar+=w*d*d; }
    const std = sw>0 ? Math.sqrt(svar/sw) : 1e-6;
    return {kind:'moment', circular:false, mean, std: std||1e-6};
  }

  function channelStat(values, weights, useHistogram, circular, domainKey){
    const Hist = global.TongMau.Histogram;
    const CS = global.TongMau.ColorSpace;
    if(useHistogram){
      const domain = circular ? {} : CHANNEL_DOMAIN[domainKey];
      const built = Hist.build(values, {weights, circular, min:domain.min, max:domain.max, bins:64});
      return Object.assign({kind:'histogram'}, built);
    }
    if(circular){
      const {mean,std} = CS.circularMeanStd(values, weights);
      return {kind:'moment', circular:true, mean, std};
    }
    return weightedMoment(values, weights);
  }

  function computeProfile(lab, options){
    const Zones = global.TongMau.Zones;
    const useLch = !!options.lch;
    const useHistogram = !!options.histogram;
    const useZone = !!options.zone;
    const {Lv, c1, c2} = extractChannels(lab, useLch);

    const profile = {
      options: {lch:useLch, histogram:useHistogram, zone:useZone},
      values: {Lv, c1, c2},
      L: channelStat(Lv, null, useHistogram, false, 'L'),
    };

    if(useZone){
      profile.zoneWeights = Zones.computeZoneWeights(Lv);
      profile.zones = Zones.ZONES.map(z=>({
        key: z.key,
        label: z.label,
        c1: channelStat(c1, profile.zoneWeights[z.key], useHistogram, false, useLch?'C':'a'),
        c2: channelStat(c2, profile.zoneWeights[z.key], useHistogram, useLch, 'b'),
      }));
    } else {
      profile.c1 = channelStat(c1, null, useHistogram, false, useLch?'C':'a');
      profile.c2 = channelStat(c2, null, useHistogram, useLch, 'b');
    }
    return profile;
  }

  function matchValue(x, ownStat, refStat){
    const CS = global.TongMau.ColorSpace;
    const Hist = global.TongMau.Histogram;
    if(ownStat.kind==='histogram'){
      return Hist.valueAtPercentile(refStat, Hist.percentileOf(ownStat, x));
    }
    if(ownStat.circular){
      const d = CS.wrapDeg180(x - ownStat.mean);
      const scaled = d * (refStat.std / (ownStat.std||1e-6));
      return CS.wrapDeg360(refStat.mean + scaled);
    }
    return (x - ownStat.mean) * ((refStat.std||1e-6) / (ownStat.std||1e-6)) + refStat.mean;
  }

  // Blends the zoned match across a pixel's own zone-membership weights.
  // Hue is blended as a unit vector (not a raw angle) so blending across
  // zones never averages e.g. 359deg and 1deg into 180deg.
  function matchZonedColor(i, c1, c2, ownZones, refZones, zoneWeights, useLch){
    let sumW=0, accC1=0, accX=0, accY=0;
    for(let z=0; z<ownZones.length; z++){
      const w = zoneWeights[ownZones[z].key][i];
      if(w<=0) continue;
      const v1 = matchValue(c1[i], ownZones[z].c1, refZones[z].c1);
      const v2 = matchValue(c2[i], ownZones[z].c2, refZones[z].c2);
      accC1 += w*v1;
      if(useLch){
        const rad = v2*Math.PI/180;
        accX += w*Math.cos(rad); accY += w*Math.sin(rad);
      } else {
        accX += w*v2;
      }
      sumW += w;
    }
    if(sumW<=0) return [c1[i], c2[i]];
    const outC1 = accC1/sumW;
    if(!useLch) return [outC1, accX/sumW];
    let hueDeg = Math.atan2(accY, accX)*180/Math.PI;
    if(hueDeg<0) hueDeg += 360;
    return [outC1, hueDeg];
  }

  // Produces a new Lab (Float32Array [L,a,b,...]) for ownProfile's pixels,
  // matched toward refProfile according to the options both profiles share.
  function apply(ownProfile, refProfile){
    const CS = global.TongMau.ColorSpace;
    const {Lv, c1, c2} = ownProfile.values;
    const n = Lv.length;
    const outLab = new Float32Array(n*3);
    const useLch = ownProfile.options.lch;
    const useZone = ownProfile.options.zone;

    for(let i=0;i<n;i++){
      const outL = matchValue(Lv[i], ownProfile.L, refProfile.L);
      let outC1, outC2;
      if(useZone){
        [outC1, outC2] = matchZonedColor(i, c1, c2, ownProfile.zones, refProfile.zones, ownProfile.zoneWeights, useLch);
      } else {
        outC1 = matchValue(c1[i], ownProfile.c1, refProfile.c1);
        outC2 = matchValue(c2[i], ownProfile.c2, refProfile.c2);
      }
      let outA, outB;
      if(useLch){
        const lab = CS.lchToLab(outL, outC1, outC2);
        outA = lab[1]; outB = lab[2];
      } else {
        outA = outC1; outB = outC2;
      }
      outLab[i*3]=outL; outLab[i*3+1]=outA; outLab[i*3+2]=outB;
    }
    return outLab;
  }

  function formatChannel(stat){
    if(stat.kind==='histogram') return 'khớp histogram (toàn bộ phân phối)';
    return `μ ${stat.mean.toFixed(1)} · σ ${stat.std.toFixed(1)}`;
  }

  // Human-readable summary for the analysis panel UI.
  function describeProfile(profile){
    const useLch = profile.options.lch;
    const c1Label = useLch ? 'C (bão hòa)' : 'a (lục–đỏ)';
    const c2Label = useLch ? 'h (tông màu °)' : 'b (lam–vàng)';
    const rows = profile.options.zone
      ? profile.zones.map(z=>({ zone: z.label, c1: formatChannel(z.c1), c2: formatChannel(z.c2) }))
      : [{ zone: null, c1: formatChannel(profile.c1), c2: formatChannel(profile.c2) }];
    return { L: formatChannel(profile.L), c1Label, c2Label, zoned: profile.options.zone, rows };
  }

  global.TongMau = global.TongMau || {};
  global.TongMau.StyleMatch = { computeProfile, apply, describeProfile };
})(window);

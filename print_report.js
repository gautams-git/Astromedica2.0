(() => {
  const escapeHtml = (value) => String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');

  const safeArray = (value) => Array.isArray(value) ? value : [];
  const yesNo = (value) => value ? 'Yes' : 'No';
  const SIGN_LORDS = {
    Aries: 'Mars', Mesha: 'Mars',
    Taurus: 'Venus', Vrishabha: 'Venus',
    Gemini: 'Mercury', Mithuna: 'Mercury',
    Cancer: 'Moon', Karka: 'Moon',
    Leo: 'Sun', Simha: 'Sun',
    Virgo: 'Mercury', Kanya: 'Mercury',
    Libra: 'Venus', Tula: 'Venus',
    Scorpio: 'Mars', Vrischika: 'Mars',
    Sagittarius: 'Jupiter', Dhanu: 'Jupiter',
    Capricorn: 'Saturn', Makara: 'Saturn',
    Aquarius: 'Saturn', Kumbha: 'Saturn',
    Pisces: 'Jupiter', Meena: 'Jupiter'
  };

  function buildRows(items, mapper) {
    return safeArray(items).map(mapper).join('');
  }

  function getPlanetDiseases(data, planetName) {
    const diseases = (data.kb_planet_diseases || {})[planetName];
    if (!diseases) return [];
    return Array.isArray(diseases) ? diseases : String(diseases).split(',').map(d => d.trim()).filter(Boolean);
  }

  function getNakshatraDiseases(data, nkName, pada) {
    if (!nkName || nkName === '—') return [];
    const nkData = (data.kb_nakshatra_diseases || {})[nkName];
    if (!nkData) return [];
    if (Array.isArray(nkData)) return nkData;
    const padaStr = String(pada || '1');
    if (nkData[padaStr]) return nkData[padaStr];
    for (const key in nkData) {
      const parts = key.split(',').map(x => x.trim());
      if (parts.includes(padaStr)) return nkData[key];
    }
    return Object.values(nkData)[0] || [];
  }

  function getRashiOrgans(data, signName) {
    const organs = (data.kb_rashi_organs || {})[signName];
    if (!organs) return [];
    return String(organs).split(',').map(o => o.trim()).filter(Boolean);
  }

  function getPlanetSign(planetObj) {
    if (!planetObj) return '—';
    const s = planetObj.sign;
    if (!s) return '—';
    if (typeof s === 'object') return s.name || '—';
    return s;
  }

  function getPlanetNakshatra(planetObj) {
    const nk = planetObj?.nakshatra;
    if (!nk) return { name: '—', pada: '' };
    if (typeof nk === 'object') return { name: nk.name || '—', pada: nk.pada || 1 };
    return { name: nk, pada: planetObj.nakshatra_pada || 1 };
  }

  function buildSummarySection(data) {
    const complete = data.complete_analysis || {};
    const diagnosis = data.diagnosis || {};
    const dasha = data.dasha || {};
    return `
      <section class="compact-summary">
        <h2>Jataka Summary</h2>
        <div class="grid two">
          <div class="card">
            <h3>Patient</h3>
            <p><strong>Name:</strong> ${escapeHtml(data.name || '—')}</p>
            <p><strong>Patient ID:</strong> ${escapeHtml(data.patient_id || '—')}</p>
            <p><strong>DOB:</strong> ${escapeHtml(data.dob || '—')}</p>
            <p><strong>Birth Time:</strong> ${escapeHtml(data.birth_time || '—')}</p>
            <p><strong>Birth Place:</strong> ${escapeHtml(data.birth_place || '—')}</p>
            <p><strong>Gender:</strong> ${escapeHtml(data.gender || '—')}</p>
          </div>
          <div class="card">
            <h3>Risk Snapshot</h3>
            <p><strong>Overall Risk:</strong> ${escapeHtml(complete.overall_risk || diagnosis.risk_score?.level || '—')}</p>
            <p><strong>Total Score:</strong> ${escapeHtml(String(complete.total_score ?? diagnosis.risk_score?.total ?? '—'))}</p>
            <p><strong>Current Mahadasha:</strong> ${escapeHtml(dasha.mahadasha?.planet || '—')}</p>
            <p><strong>Current Antardasha:</strong> ${escapeHtml(dasha.antardasha?.planet || '—')}</p>
            <p><strong>Current Pratyantardasha:</strong> ${escapeHtml(dasha.pratyantardasha?.planet || '—')}</p>
          </div>
        </div>
      </section>
    `;
  }

  function buildChartSection(data) {
    const chartMarkup = data.chart_png
      ? `<img src="${escapeHtml(data.chart_png)}" alt="Jataka Chart">`
      : (data.svg_raw || '');
    if (!chartMarkup) return '';
    return `
      <section>
        <h2>Rasi Chart</h2>
        <div class="chart-card">
          ${chartMarkup}
        </div>
      </section>
    `;
  }

  function formatPlanetLine(p) {
    if (!p || !p.name) return '—';
    return `${p.name} · House ${p.house ?? '—'} · ${p.sign || '—'} · ${p.degree ?? '—'}° ${p.minutes ?? 0}' · ${p.nakshatra || '—'} P${p.nakshatra_pada || '—'}`;
  }

  function buildRule1Section(data) {
    const r = data.rule1 || {};
    if (!Object.keys(r).length) return '';
    const rk1 = safeArray(r.rogkaraka_1);
    const eighthOccupants = safeArray(r.eighth_house_occupants);
    const drishti = safeArray(r.drishti_on_6th || r.complete_analysis?.drishti?.aspects_sixth_house);
    const moonObj = safeArray(data.planets).find(p => p.name === 'Moon') || null;
    const moonNk = moonObj ? (moonObj.nakshatra || {}) : {};
    const moonNkName = moonObj ? (typeof moonNk === 'object' ? (moonNk.name || '—') : (moonNk || '—')) : '—';
    const moonNkPada = moonObj ? (typeof moonNk === 'object' ? (moonNk.pada || '') : (moonObj.nakshatra_pada || '')) : '';
    const moonHouseSign = moonObj
      ? (((safeArray(data.houses).find(h => h.house === moonObj.house)?.sign || {}).name) || '—')
      : '—';
    const moonHouseLord = SIGN_LORDS[moonHouseSign] || '—';
    const moonHouseOccupants = moonObj
      ? safeArray(data.planets).filter(p => p.house === moonObj.house && p.name !== 'Moon' && p.name !== 'Ascendant').map(p => p.name)
      : [];
    return `
      <section>
        <h2>Rule 1 Full Box</h2>
        <div class="grid two">
          <div class="card">
            <h3>Core Houses</h3>
            <p><strong>Ascendant House:</strong> ${escapeHtml(String(r.ascendant_house ?? '—'))}</p>
            <p><strong>Ascendant Sign:</strong> ${escapeHtml(r.ascendant_sign || '—')}</p>
            <p><strong>6th House:</strong> ${escapeHtml(String(r.sixth_house_number ?? '—'))} (${escapeHtml(r.sixth_house_sign || '—')})</p>
            <p><strong>6th House Lord:</strong> ${escapeHtml(r.sixth_house_lord || '—')}</p>
            <p><strong>8th House:</strong> ${escapeHtml(String(r.eighth_house_number ?? '—'))} (${escapeHtml(r.eighth_house_sign || '—')})</p>
            <p><strong>8th House Lord:</strong> ${escapeHtml(r.eighth_house_lord || '—')}</p>
          </div>
          <div class="card">
            <h3>Priority Summary</h3>
            <p><strong>RK1 Occupants:</strong> ${escapeHtml(rk1.map(p => p.name).join(', ') || 'None')}</p>
            <p><strong>RK2 6th Lord:</strong> ${escapeHtml(r.rogkaraka_2?.name || 'None')}</p>
            <p><strong>RK3 Dispositor:</strong> ${escapeHtml(r.rogkaraka_3?.name || 'None')}</p>
            <p><strong>8th Occupants:</strong> ${escapeHtml(eighthOccupants.map(p => p.name).join(', ') || 'None')}</p>
            <p><strong>Drishti on 6th:</strong> ${escapeHtml(drishti.map(p => p.planet || p.name).join(', ') || 'None')}</p>
          </div>
        </div>
        <div class="card">
          <h3>RK1 - 6th House Occupants</h3>
          <ul>${rk1.length ? rk1.map(p => `<li>${escapeHtml(formatPlanetLine(p))}</li>`).join('') : '<li>None</li>'}</ul>
        </div>
        <div class="card">
          <h3>Moon Position</h3>
          ${moonObj ? `
            <p><strong>Moon Placement:</strong> House ${escapeHtml(String(moonObj.house ?? '—'))} (${escapeHtml(moonObj.sign?.name || '—')}) at ${escapeHtml(String(moonObj.degree ?? '—'))}° ${escapeHtml(String(moonObj.minutes ?? 0))}'</p>
            <p><strong>Nakshatra:</strong> ${escapeHtml(moonNkName)}${moonNkPada ? ` (Pada ${escapeHtml(String(moonNkPada))})` : ''}</p>
            <p><strong>House Lord Check:</strong> Moon is in House ${escapeHtml(String(moonObj.house ?? '—'))}, whose sign is ${escapeHtml(moonHouseSign)}, so the house lord is ${escapeHtml(moonHouseLord)}.</p>
            <p><strong>House Occupants:</strong> ${escapeHtml(moonHouseOccupants.join(', ') || 'No other planet is sitting with Moon in this house.')}</p>
          ` : `
            <p>Moon position is not available in this chart payload.</p>
          `}
        </div>
        <div class="grid two">
          <div class="card">
            <h3>RK2 - 6th House Lord</h3>
            <p>${escapeHtml(formatPlanetLine(r.rogkaraka_2))}</p>
          </div>
          <div class="card">
            <h3>RK3 - Dispositor Chain</h3>
            <p>${escapeHtml(formatPlanetLine(r.rogkaraka_3))}</p>
          </div>
        </div>
        <div class="card">
          <h3>8th House Chronic / Surgery Indicators</h3>
          <ul>${eighthOccupants.length ? eighthOccupants.map(p => `<li>${escapeHtml(formatPlanetLine(p))}</li>`).join('') : '<li>No 8th house occupant captured</li>'}</ul>
        </div>
      </section>
    `;
  }

  function extractHitlistOrgansAndDiseases(sys) {
    const organs = new Set();
    const diseases = new Set();
    safeArray(sys.trail).forEach(t => {
      const source = t.source || '';
      if (source.includes('HL3')) organs.add(t.term);
      if (source.includes('HL2')) diseases.add(t.term);
    });
    return {
      organs: Array.from(organs),
      diseases: Array.from(diseases),
    };
  }

  function buildMostProbableSection(data) {
    const hitlist = data.hitlist || {};
    const systems = safeArray(hitlist.top4 || hitlist.scored_systems).slice(0, 4);
    const topDiseases = safeArray(data.top_diseases || data.complete_analysis?.top_diseases);
    const mostProbable = safeArray(data.most_probable);
    return `
      <section>
        <h2>Most Probable Organs And Diseases</h2>
        <div class="grid four">
          ${systems.map(sys => {
            const extracted = extractHitlistOrgansAndDiseases(sys);
            return `
              <div class="card priority-card">
                <div class="score-badge">${escapeHtml(Number(sys.score || 0).toFixed(1))}</div>
                <h3>${escapeHtml(sys.label || sys.display_name || '—')}</h3>
                <p class="muted-line">${escapeHtml(sys.reason || `${safeArray(sys.hitlists).join(' + ')} Agreement`)}</p>
                <div class="mini-block">
                  <div class="mini-title">Organs At Risk</div>
                  <div class="pill-wrap">${(extracted.organs.length ? extracted.organs : ['None']).map(item => `<span class="pill organ-pill">${escapeHtml(item)}</span>`).join('')}</div>
                </div>
                <div class="mini-block">
                  <div class="mini-title">Most Probable Diseases</div>
                  <div class="pill-wrap">${(extracted.diseases.length ? extracted.diseases : ['None']).map(item => `<span class="pill disease-pill">${escapeHtml(item)}</span>`).join('')}</div>
                </div>
              </div>
            `;
          }).join('')}
        </div>
        <div class="grid two">
          <div class="card">
            <h3>Top Diseases</h3>
            <ul>${topDiseases.length ? topDiseases.map(item => `<li>${escapeHtml(item.disease || item.condition || JSON.stringify(item))}</li>`).join('') : '<li>None</li>'}</ul>
          </div>
          <div class="card">
            <h3>Most Probable List</h3>
            <ul>${mostProbable.length ? mostProbable.map(item => `<li>${escapeHtml(typeof item === 'string' ? item : (item.label || item.condition || JSON.stringify(item)))}</li>`).join('') : '<li>None</li>'}</ul>
          </div>
        </div>
      </section>
    `;
  }

  function buildZoneSection(data) {
    const z = data.zone_analysis || {};
    if (!Object.keys(z).length) return '';
    return `
      <section>
        <h2>RL / Zone Score</h2>
        <div class="grid three">
          <div class="card">
            <h3>Array 1</h3>
            <p><strong>Asc Lord + 8th Lord:</strong> ${escapeHtml(z.array1_display || '—')}</p>
            <p><strong>Score:</strong> ${escapeHtml(String(z.array1_score ?? '—'))}</p>
          </div>
          <div class="card">
            <h3>Array 2</h3>
            <p><strong>Moon + Saturn:</strong> ${escapeHtml(z.array2_display || '—')}</p>
            <p><strong>Score:</strong> ${escapeHtml(String(z.array2_score ?? '—'))}</p>
          </div>
          <div class="card">
            <h3>Array 3</h3>
            <p><strong>Asc + 2nd Asc:</strong> ${escapeHtml(z.array3_display || '—')}</p>
            <p><strong>Score:</strong> ${escapeHtml(String(z.array3_score ?? '—'))}</p>
          </div>
        </div>
        <div class="card">
          <h3>Total Zone Result</h3>
          <p><strong>Total Score:</strong> ${escapeHtml(String(z.score ?? '—'))}</p>
          <p><strong>Rule Match:</strong> ${escapeHtml(z.matched_rule || '—')}</p>
        </div>
      </section>
    `;
  }

  function buildCurrentDashaSection(data) {
    const dasha = data.dasha || {};
    const peakRiskItems = safeArray(data.active_disease_indicators || data.complete_analysis?.active_disease_indicators);
    return `
      <section>
        <h2>Current Dasha Period</h2>
        <div class="grid three">
          <div class="card">
            <h3>Mahadasha</h3>
            <p><strong>Planet:</strong> ${escapeHtml(dasha.mahadasha?.planet || '—')}</p>
            <p><strong>Start:</strong> ${escapeHtml(dasha.mahadasha?.start_date || '—')}</p>
            <p><strong>End:</strong> ${escapeHtml(dasha.mahadasha?.end_date || '—')}</p>
          </div>
          <div class="card">
            <h3>Antardasha</h3>
            <p><strong>Planet:</strong> ${escapeHtml(dasha.antardasha?.planet || '—')}</p>
            <p><strong>Start:</strong> ${escapeHtml(dasha.antardasha?.start_date || '—')}</p>
            <p><strong>End:</strong> ${escapeHtml(dasha.antardasha?.end_date || '—')}</p>
          </div>
          <div class="card">
            <h3>Pratyantardasha</h3>
            <p><strong>Planet:</strong> ${escapeHtml(dasha.pratyantardasha?.planet || '—')}</p>
            <p><strong>Start:</strong> ${escapeHtml(dasha.pratyantardasha?.start_date || '—')}</p>
            <p><strong>End:</strong> ${escapeHtml(dasha.pratyantardasha?.end_date || '—')}</p>
          </div>
        </div>
        ${peakRiskItems.length ? `
          <div class="card">
            <h3>Peak Risk Period Detected</h3>
            <ul>${peakRiskItems.map(item => `<li>${escapeHtml(typeof item === 'string' ? item : (item.message || item.label || JSON.stringify(item)))}</li>`).join('')}</ul>
          </div>
        ` : ''}
      </section>
    `;
  }

  function buildPastFiveYearsSection(data) {
    const rows = safeArray(data.dasha?.past_5_years).slice(0, 5);
    if (!rows.length) return '';
    return `
      <section>
        <h2>Past 5 Years Dasha Timeline</h2>
        <table>
          <thead>
            <tr>
              <th>Mahadasha</th>
              <th>Antardasha</th>
              <th>Pratyantardasha</th>
              <th>Start</th>
              <th>End</th>
            </tr>
          </thead>
          <tbody>
            ${buildRows(rows, item => `
              <tr>
                <td>${escapeHtml(item.mahadasha || '—')}</td>
                <td>${escapeHtml(item.antardasha || '—')}</td>
                <td>${escapeHtml(item.pratyantardasha || '—')}</td>
                <td>${escapeHtml(item.start_date || '—')}</td>
                <td>${escapeHtml(item.end_date || '—')}</td>
              </tr>
            `)}
          </tbody>
        </table>
      </section>
    `;
  }

  function buildExpandedDashaHistorySection(data) {
    const rows = safeArray(data.dasha?.expanded_history);
    if (!rows.length) return '';
    return `
      <section>
        <details class="card dasha-history-dropdown">
          <summary>Show Full Dasha History For Current + Previous Mahadasha</summary>
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Mahadasha</th>
                  <th>Antardasha</th>
                  <th>Pratyantardasha</th>
                  <th>Start</th>
                  <th>End</th>
                </tr>
              </thead>
              <tbody>
                ${buildRows(rows, item => `
                  <tr>
                    <td>${escapeHtml(item.mahadasha || '—')}</td>
                    <td>${escapeHtml(item.antardasha || '—')}</td>
                    <td>${escapeHtml(item.pratyantardasha || '—')}</td>
                    <td>${escapeHtml(item.start_date || '—')}</td>
                    <td>${escapeHtml(item.end_date || '—')}</td>
                  </tr>
                `)}
              </tbody>
            </table>
          </div>
        </details>
      </section>
    `;
  }

  function buildHitlistSection(data) {
    const hitlist = data.hitlist || {};
    const systems = safeArray(hitlist.top4 || hitlist.scored_systems);
    if (!systems.length) return '';
    const maxScore = Math.max(...systems.map(sys => sys.score || 0), 10);
    return `
      <section>
        <div class="hitlist-card">
          <div class="hitlist-header">
            <div class="hitlist-title">System Priority (HitList)</div>
            <div class="print-note" style="text-transform:uppercase; letter-spacing:0.08em;">Weighted Multi-Layer Priority</div>
          </div>
          <div class="hitlist-grid">
            ${systems.map(sys => {
              const label = sys.label || sys.display_name || '—';
              const reason = sys.reason || `${safeArray(sys.hitlists).join(' + ')} Agreement (Convergence: ${sys.convergence || 1})`;
              const progress = ((sys.score || 0) / maxScore) * 100;
              const extracted = extractHitlistOrgansAndDiseases(sys);
              return `
                <div class="hit-item">
                  <div class="hit-score-badge">${escapeHtml(Number(sys.score || 0).toFixed(1))}</div>
                  <div class="hit-system">${escapeHtml(label)}</div>
                  <div class="hit-reason">${escapeHtml(reason)}</div>
                  <div class="hit-progress-container">
                    <div class="hit-progress-fill" style="width:${progress}%"></div>
                  </div>
                  <div class="hit-divider"></div>
                  <div class="hit-details">
                    ${extracted.organs.length ? `<div class="hit-detail-section"><span class="hit-detail-label">Organs at Risk</span><div class="hit-tags">${extracted.organs.slice(0, 6).map(item => `<span class="hit-tag organ">${escapeHtml(item)}</span>`).join('')}</div></div>` : ''}
                    ${extracted.diseases.length ? `<div class="hit-detail-section"><span class="hit-detail-label">Most Probable Diseases</span><div class="hit-tags">${extracted.diseases.slice(0, 6).map(item => `<span class="hit-tag disease">${escapeHtml(item)}</span>`).join('')}</div></div>` : ''}
                  </div>
                  <div class="hit-footer">
                    <span>Match: ${escapeHtml(String(sys.convergence ?? '—'))} Lists</span>
                    <span class="hit-analyze">Analyze Logic</span>
                  </div>
                </div>
              `;
            }).join('')}
          </div>
        </div>
      </section>
    `;
  }

  function buildDiseaseTableSection(data) {
    const rows = safeArray(data.complete_analysis?.disease_risks);
    if (!rows.length) return '';
    return `
      <section class="compact-disease-table">
        <h2>Complete Disease Table</h2>
        <div class="disease-table-wrap">
          ${rows.map(item => {
            const risk = item.risk_level || 'LOW';
            const riskClass = risk === 'CRITICAL' ? 'risk-critical' : risk === 'HIGH' ? 'risk-high' : risk === 'MODERATE' ? 'risk-moderate' : 'risk-low';
            return `
              <div class="disease-row ${riskClass}">
                <div class="disease-row-main">
                  <div>
                    <div class="disease-name">${escapeHtml(item.condition || '—')}</div>
                    <div class="disease-source">${escapeHtml(item.category || '—')} · ${escapeHtml(item.source || '—')}</div>
                  </div>
                  <div class="disease-badge ${riskClass}">${escapeHtml(risk)}</div>
                </div>
                <div class="disease-scoreline">
                  <span>Base: ${escapeHtml(String(item.base_score ?? '—'))}</span>
                  <span>Drishti: ${escapeHtml(String(item.drishti_bonus ?? '—'))}</span>
                  <span>Dasha: ${escapeHtml(String(item.dasha_bonus ?? '—'))}</span>
                  <span>Total: ${escapeHtml(String(item.total_score ?? '—'))}</span>
                </div>
              </div>
            `;
          }).join('')}
        </div>
      </section>
    `;
  }

  function buildReferenceTableSection(data) {
    const rows = [];
    const occupants = data.rule1?.rogkaraka_1 || [];
    const hasP1 = occupants.length > 0;
    const dispositor = data.rule1?.rogkaraka_3;

    if (!hasP1 && dispositor) {
      const nk = getPlanetNakshatra(dispositor);
      rows.push({
        priority: 'P2',
        label: 'Dispositor · of 6th Lord',
        planet: dispositor.name,
        planetDiseases: getPlanetDiseases(data, dispositor.name),
        nakshatra: nk.name,
        pada: nk.pada,
        nakshatraDiseases: getNakshatraDiseases(data, nk.name, nk.pada),
        rashi: '—',
        rashiOrgans: [],
        rashiApplicable: false
      });
    }

    occupants.forEach(p => {
      const nk = getPlanetNakshatra(p);
      const sign = getPlanetSign(p);
      rows.push({
        priority: 'P1',
        label: 'Occupant · 6th House',
        planet: p.name,
        planetDiseases: getPlanetDiseases(data, p.name),
        nakshatra: nk.name,
        pada: nk.pada,
        nakshatraDiseases: getNakshatraDiseases(data, nk.name, nk.pada),
        rashi: sign,
        rashiOrgans: getRashiOrgans(data, sign),
        rashiApplicable: true
      });
    });

    if (hasP1) {
      if (dispositor) {
        const nk = getPlanetNakshatra(dispositor);
        rows.push({
          priority: 'P2',
          label: 'Dispositor · of 6th Lord',
          planet: dispositor.name,
          planetDiseases: getPlanetDiseases(data, dispositor.name),
          nakshatra: nk.name,
          pada: nk.pada,
          nakshatraDiseases: getNakshatraDiseases(data, nk.name, nk.pada),
          rashi: '—',
          rashiOrgans: [],
          rashiApplicable: false
        });
      } else {
        rows.push({
          priority: 'P2',
          label: 'Dispositor · Not Applicable',
          planet: '—',
          planetDiseases: [],
          nakshatra: '—',
          pada: '',
          nakshatraDiseases: [],
          rashi: '—',
          rashiOrgans: [],
          rashiApplicable: false
        });
      }
    }

    const drishti = data.complete_analysis?.drishti?.aspects_sixth_house || [];
    if (drishti.length > 0) {
      drishti.forEach(asp => {
        const pData = safeArray(data.planets).find(p => p.name === asp.planet);
        if (!pData) return;
        const nk = getPlanetNakshatra(pData);
        rows.push({
          priority: 'P3',
          label: `Drishti · from House ${asp.from_house}`,
          planet: asp.planet,
          planetDiseases: getPlanetDiseases(data, asp.planet),
          nakshatra: nk.name,
          pada: nk.pada,
          nakshatraDiseases: getNakshatraDiseases(data, nk.name, nk.pada),
          rashi: '—',
          rashiOrgans: [],
          rashiApplicable: false
        });
      });
    } else {
      rows.push({
        priority: 'P3',
        label: 'Drishti · None',
        planet: '—',
        planetDiseases: [],
        nakshatra: '—',
        pada: '',
        nakshatraDiseases: [],
        rashi: '—',
        rashiOrgans: [],
        rashiApplicable: false
      });
    }

    const rk2 = data.rule1?.rogkaraka_2;
    if (rk2) {
      const nk = getPlanetNakshatra(rk2);
      rows.push({
        priority: 'P4',
        label: '6th House Lord',
        planet: rk2.name,
        planetDiseases: getPlanetDiseases(data, rk2.name),
        nakshatra: nk.name,
        pada: nk.pada,
        nakshatraDiseases: getNakshatraDiseases(data, nk.name, nk.pada),
        rashi: '—',
        rashiOrgans: [],
        rashiApplicable: false
      });
    }

    const conditionalMoon = data.rule1?.conditional_moon;
    if (conditionalMoon && conditionalMoon.include && conditionalMoon.planet) {
      const moonData = conditionalMoon.planet;
      const nk = getPlanetNakshatra(moonData);
      const sign = getPlanetSign(moonData);
      rows.push({
        priority: 'PM',
        label: `Moon · via ${String(conditionalMoon.connected_role || 'link').replaceAll('_', ' ')}`,
        planet: moonData.name,
        planetDiseases: getPlanetDiseases(data, moonData.name),
        nakshatra: nk.name,
        pada: nk.pada,
        nakshatraDiseases: getNakshatraDiseases(data, nk.name, nk.pada),
        rashi: sign,
        rashiOrgans: getRashiOrgans(data, sign),
        rashiApplicable: true
      });
    }

    const ascSign = data.ascendant?.name || '';
    const ascLordName = SIGN_LORDS[ascSign] || '';
    if (ascLordName) {
      const ascLordData = safeArray(data.planets).find(p => p.name === ascLordName);
      if (ascLordData) {
        const nk = getPlanetNakshatra(ascLordData);
        const sign = getPlanetSign(ascLordData);
        rows.push({
          priority: 'P5',
          label: 'Ascendant Lord',
          planet: ascLordName,
          planetDiseases: getPlanetDiseases(data, ascLordName),
          nakshatra: nk.name,
          pada: nk.pada,
          nakshatraDiseases: getNakshatraDiseases(data, nk.name, nk.pada),
          rashi: sign,
          rashiOrgans: getRashiOrgans(data, sign),
          rashiApplicable: true
        });
      }
    }

    const sixthSign = data.rule1?.sixth_house_sign;
    if (sixthSign) {
      rows.push({
        priority: 'P6',
        label: '6th House Rashi',
        planet: '—',
        planetDiseases: [],
        nakshatra: '—',
        pada: '',
        nakshatraDiseases: [],
        rashi: sixthSign,
        rashiOrgans: getRashiOrgans(data, sixthSign),
        rashiApplicable: true
      });
    }

    if (ascSign) {
      rows.push({
        priority: 'P7',
        label: 'Ascendant Rashi',
        planet: '—',
        planetDiseases: [],
        nakshatra: '—',
        pada: '',
        nakshatraDiseases: [],
        rashi: ascSign,
        rashiOrgans: getRashiOrgans(data, ascSign),
        rashiApplicable: true
      });
    }

    const priorityColorMap = {
      P1: '#ef4444',
      P2: '#f59e0b',
      P3: '#a78bfa',
      P4: '#60a5fa',
      PM: '#38bdf8',
      P5: '#34d399',
      P6: '#9a6f00',
      P7: '#f472b6'
    };

    return `
      <section>
        <h2>Complete Disease Reference Table</h2>
        <div class="reference-note">All sources · Unfiltered · For manual review</div>
        <div class="reference-table-wrap">
          ${rows.map(row => {
            const color = priorityColorMap[row.priority] || '#9a6f00';
            return `
              <div class="reference-block">
                <div class="reference-header" style="background:${color}18; border-left-color:${color};">
                  <span class="reference-priority" style="color:${color}; background:${color}22; border-color:${color}44;">${escapeHtml(row.priority)}</span>
                  <span class="reference-label">${escapeHtml(row.label)}</span>
                  <span class="reference-meta">· ${escapeHtml(row.planet)} ${row.nakshatra !== '—' ? `· ${escapeHtml(row.nakshatra)} Pada ${escapeHtml(String(row.pada))}` : ''} ${row.rashi !== '—' ? `· ${escapeHtml(row.rashi)}` : ''}</span>
                </div>
                <div class="reference-grid">
                  <div class="reference-col">
                    <div class="reference-title" style="color:${color};">Planet Organs ${row.planet !== '—' ? `· ${escapeHtml(row.planet)}` : ''}</div>
                    ${row.planetDiseases.length ? row.planetDiseases.map(item => `<div class="reference-item"><span style="color:${color};">›</span>${escapeHtml(item)}</div>`).join('') : '<div class="reference-empty">—</div>'}
                  </div>
                  <div class="reference-col reference-col-alt">
                    <div class="reference-title nakshatra-title">Nakshatra Diseases ${row.nakshatra !== '—' ? `· ${escapeHtml(row.nakshatra)} P${escapeHtml(String(row.pada))}` : ''}</div>
                    ${row.nakshatraDiseases.length ? row.nakshatraDiseases.map(item => `<div class="reference-item"><span class="nakshatra-dot">›</span>${escapeHtml(item)}</div>`).join('') : '<div class="reference-empty">—</div>'}
                  </div>
                  <div class="reference-col">
                    <div class="reference-title rashi-title">Rashi Organs ${row.rashi !== '—' ? `· ${escapeHtml(row.rashi)}` : ''}</div>
                    ${!row.rashiApplicable ? '<div class="reference-empty">Not Applicable</div>' : row.rashiOrgans.length ? row.rashiOrgans.map(item => `<div class="reference-item"><span class="rashi-dot">›</span>${escapeHtml(item)}</div>`).join('') : '<div class="reference-empty">—</div>'}
                  </div>
                </div>
              </div>
            `;
          }).join('')}
        </div>
        <div class="reference-disclaimer">
          This table shows all raw astrological data unfiltered. Every disease and organ listed here is a classical reference, not a confirmed diagnosis. Use this for manual cross-referencing only.
        </div>
      </section>
    `;
  }

  function buildRashiCorrelationSection(data) {
    const rc = data.rashi_correlation || {};
    const top3 = safeArray(rc.top3);
    if (!top3.length) return '';
    const maxScore = Math.max(...top3.map(item => item.score || 0), 10);
    const anchors = rc.anchors || {};
    const anchorSummary = [
      anchors.P6 ? `P6: ${anchors.P6.sign || '-'} (${safeArray(anchors.P6.terms).slice(0, 5).join(', ') || 'No organs'})` : '',
      anchors.P7 ? `P7: ${anchors.P7.sign || '-'} (${safeArray(anchors.P7.terms).slice(0, 5).join(', ') || 'No organs'})` : ''
    ].filter(Boolean).join(' | ');
    return `
      <section>
        <div class="hitlist-card" style="border-top-color:#3b82f6;">
          <div class="hitlist-header">
            <div>
              <div class="hitlist-title" style="color:#3b82f6;">Rashi Priority Correlation</div>
              <div class="print-note">New independent box. Anchors all matching to P6 (6th house rashi) and P7 (ascendant rashi) without changing the current HitList.</div>
            </div>
            <div class="print-note" style="text-transform:uppercase; letter-spacing:0.08em;">Top 3 Rashi-Anchored Clusters</div>
          </div>
          ${anchorSummary ? `<div class="print-note" style="margin-bottom:10px;"><strong style="color:#9dc0ff;">Backend Anchors:</strong> ${escapeHtml(anchorSummary)}</div>` : ''}
          <div class="hitlist-grid">
            ${top3.map((item, index) => {
              const progress = ((item.score || 0) / maxScore) * 100;
              const matchedAnchors = safeArray(item.matched_anchors);
              const priorities = safeArray(item.supporting_priorities);
              const organs = safeArray(item.matched_organs).slice(0, 6);
              const diseases = safeArray(item.matched_diseases).slice(0, 6);
              return `
                <div class="hit-item">
                  <div class="hit-score-badge" style="background:#eef5ff; color:#3b82f6; border-color:#9abcf5;">${escapeHtml(Number(item.score || 0).toFixed(1))}</div>
                  <div class="print-note" style="color:#9dc0ff; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:5px;">Top ${index + 1} Match</div>
                  <div class="hit-system">${escapeHtml(item.label || '—')}</div>
                  <div class="hit-reason">${escapeHtml(item.reason || '—')}</div>
                  <div class="hit-progress-container">
                    <div class="hit-progress-fill" style="width:${progress}%; background:linear-gradient(90deg, #3b82f6, #60a5fa);"></div>
                  </div>
                  <div class="hit-divider"></div>
                  <div class="hit-details">
                    <div class="hit-detail-section">
                      <span class="hit-detail-label">Matched Anchors</span>
                      <div class="hit-tags">${matchedAnchors.map(v => `<span class="hit-tag organ">${escapeHtml(v)}</span>`).join('')}</div>
                    </div>
                    <div class="hit-detail-section">
                      <span class="hit-detail-label">Supporting Priorities</span>
                      <div class="hit-tags">${priorities.map(v => `<span class="hit-tag disease">${escapeHtml(v)}</span>`).join('')}</div>
                    </div>
                    ${organs.length ? `<div class="hit-detail-section"><span class="hit-detail-label">Rashi Organs Matched</span><div class="hit-tags">${organs.map(v => `<span class="hit-tag organ">${escapeHtml(v)}</span>`).join('')}</div></div>` : ''}
                    ${diseases.length ? `<div class="hit-detail-section"><span class="hit-detail-label">Evidence Terms Matched</span><div class="hit-tags">${diseases.map(v => `<span class="hit-tag disease">${escapeHtml(v)}</span>`).join('')}</div></div>` : ''}
                  </div>
                  <div class="hit-footer">
                    <span>Top ${escapeHtml(String(index + 1))} Match</span>
                    <span class="hit-analyze">Analyze Logic</span>
                  </div>
                </div>
              `;
            }).join('')}
          </div>
        </div>
      </section>
    `;
  }

  function buildCurrentPlanetsSection(data) {
    const rows = safeArray(data.current_planetary_positions?.planets).filter(p => p.name !== 'Ascendant');
    if (!rows.length) return '';
    return `
      <section>
        <h2>Current Planet Position Impact</h2>
        <table>
          <thead>
            <tr>
              <th>Planet</th>
              <th>House</th>
              <th>Sign</th>
              <th>Degree</th>
              <th>Nakshatra</th>
              <th>Retrograde</th>
            </tr>
          </thead>
          <tbody>
            ${buildRows(rows, item => {
              const nk = item.nakshatra || {};
              const nkName = typeof nk === 'object' ? nk.name : nk;
              const nkPada = typeof nk === 'object' ? nk.pada : item.nakshatra_pada;
              return `
                <tr>
                  <td>${escapeHtml(item.name || '—')}</td>
                  <td>${escapeHtml(String(item.absolute_house || item.house || '—'))}</td>
                  <td>${escapeHtml(item.sign?.name || '—')}</td>
                  <td>${escapeHtml(String(item.degree ?? '—'))}° ${escapeHtml(String(item.minutes ?? 0))}'</td>
                  <td>${escapeHtml(nkName || '—')}${nkPada ? ` (P${escapeHtml(String(nkPada))})` : ''}</td>
                  <td>${escapeHtml(yesNo(item.isRetrograde))}</td>
                </tr>
              `;
            })}
          </tbody>
        </table>
      </section>
    `;
  }

  function buildDistanceSections(data) {
    const impRows = safeArray(data.imp_rashi_distance?.rows);
    const birthRows = safeArray(data.birth_current_distance?.rows);
    if (!impRows.length && !birthRows.length) return '';

    const birthByRole = new Map(birthRows.map(row => [row.role, row]));
    const verdictRows = impRows.map(impRow => {
      const birthRow = birthByRole.get(impRow.role);
      if (!birthRow) return null;
      let verdict = 'Insufficient Data';
      if (impRow.distance_bucket === 'increase' && birthRow.distance_bucket === 'increase') verdict = 'Increased Severity';
      else if (impRow.distance_bucket === 'lower' && birthRow.distance_bucket === 'lower') verdict = 'Reduced Impact';
      else if (
        (impRow.distance_bucket === 'increase' && birthRow.distance_bucket === 'lower') ||
        (impRow.distance_bucket === 'lower' && birthRow.distance_bucket === 'increase')
      ) verdict = 'Moderate / Mixed Impact';
      return { impRow, birthRow, verdict };
    }).filter(Boolean);

    return `
      <section>
        <h2>IMP Rashi Distance</h2>
        <table>
          <thead>
            <tr>
              <th>Role</th>
              <th>Planet</th>
              <th>Distance</th>
              <th>Group</th>
              <th>Effect</th>
            </tr>
          </thead>
          <tbody>
            ${buildRows(impRows, row => `
              <tr>
                <td>${escapeHtml(row.role || '—')}</td>
                <td>${escapeHtml(row.planet || '—')}</td>
                <td>${escapeHtml(String(row.distance ?? '—'))}</td>
                <td>${escapeHtml(row.distance_bucket || '—')}</td>
                <td>${escapeHtml(row.effect || '—')}</td>
              </tr>
            `)}
          </tbody>
        </table>
      </section>
      <section>
        <h2>Planet Distance</h2>
        <table>
          <thead>
            <tr>
              <th>Role</th>
              <th>Planet</th>
              <th>Distance</th>
              <th>Group</th>
              <th>Effect</th>
            </tr>
          </thead>
          <tbody>
            ${buildRows(birthRows, row => `
              <tr>
                <td>${escapeHtml(row.role || '—')}</td>
                <td>${escapeHtml(row.planet || '—')}</td>
                <td>${escapeHtml(String(row.distance ?? '—'))}</td>
                <td>${escapeHtml(row.distance_bucket || '—')}</td>
                <td>${escapeHtml(row.effect || '—')}</td>
              </tr>
            `)}
          </tbody>
        </table>
      </section>
      <section>
        <h2>Combined Distance Verdict</h2>
        <table>
          <thead>
            <tr>
              <th>Role</th>
              <th>Planet</th>
              <th>IMP Rashi</th>
              <th>Planet Distance</th>
              <th>Verdict</th>
            </tr>
          </thead>
          <tbody>
            ${buildRows(verdictRows, row => `
              <tr>
                <td>${escapeHtml(row.impRow.role || '—')}</td>
                <td>${escapeHtml(row.impRow.planet || row.birthRow.planet || '—')}</td>
                <td>${escapeHtml(String(row.impRow.distance ?? '—'))} (${escapeHtml(row.impRow.distance_bucket || '—')})</td>
                <td>${escapeHtml(String(row.birthRow.distance ?? '—'))} (${escapeHtml(row.birthRow.distance_bucket || '—')})</td>
                <td>${escapeHtml(row.verdict)}</td>
              </tr>
            `)}
          </tbody>
        </table>
      </section>
    `;
  }


  function buildSimpleZoneSection(data) {
    const z = data.zone_analysis || {};
    if (!Object.keys(z).length) return '';
    const totalScore = Number(z.score ?? 0);
    const zoneResultMap = [
      { total: 15, result: 2 },
      { total: 17, result: 2.2 },
      { total: 20, result: 2 },
      { total: 24, result: 3 },
      { total: 19, result: 3.25 },
      { total: 21, result: 4 },
      { total: 25, result: 3.2 },
      { total: 27, result: 3.3 },
      { total: 30, result: 3.5 }
    ];
    const matchedZoneResult = zoneResultMap.find(row => row.total === totalScore);
    const relativeScore = z.relative_score ?? z.relative_marks ?? matchedZoneResult?.result ?? '—';
    return `
      <section>
        <h2>Zone Score</h2>
        <div class="zone-summary">
          <div class="zone-score-card">
            <div class="label">Zone Score</div>
            <div class="value">${escapeHtml(String(z.score ?? '—'))}</div>
          </div>
          <div class="zone-score-card">
            <div class="label">Relative Score</div>
            <div class="value">${escapeHtml(String(relativeScore))}</div>
          </div>
        </div>
      </section>
    `;
  }

  function buildPrintableReportBody(data) {
    return `
      <header>
        <h1>AstroMedica Offline Summary</h1>
        <div class="meta">Generated for print/PDF export</div>
      </header>
      ${buildSummarySection(data)}
      ${buildChartSection(data)}
      ${buildRule1Section(data)}
      ${buildCurrentDashaSection(data)}
      ${buildPastFiveYearsSection(data)}
      ${buildExpandedDashaHistorySection(data)}
      ${buildHitlistSection(data)}
      ${buildRashiCorrelationSection(data)}
      ${buildCurrentPlanetsSection(data)}
      ${buildSimpleZoneSection(data)}
      ${buildDistanceSections(data)}
    `;
  }

  function getStoredChartData() {
    try {
      return JSON.parse(sessionStorage.getItem('chartData') || '{}');
    } catch (err) {
      console.error('Could not parse chartData from sessionStorage', err);
      return {};
    }
  }

  window.renderPrintableReportPage = function renderPrintableReportPage(options = {}) {
    const data = options.data || getStoredChartData();
    const mount = document.getElementById('print-report-root') || document.body;
    mount.innerHTML = buildPrintableReportBody(data);
    if (options.autoPrint) {
      window.setTimeout(() => window.print(), 250);
    }
  };

  window.navigateToPrintableReport = function navigateToPrintableReport() {
    window.location.href = '/print-report';
  };

  window.openPrintableReport = window.navigateToPrintableReport;
})();

import csv
contexts = [
  {ticker: 'ACE'},
  {ticker: 'SHR'},
  {ticker: 'RBF'},
  {ticker: 'CPW'},
  {ticker: 'AWC'},
  {ticker: 'DOHOME'},
  {ticker: 'SEG'},
  {ticker: 'ILM'},
  {ticker: 'TTT'},
  {ticker: 'VRANDA'},
  {ticker: 'ZEN'},
  {ticker:'TQM'},
  {ticker:'NER'},
  {ticker:'PR9'},
  {ticker:'BGC'},
  {ticker:'OSP'},
  {ticker:'COTTO'},
  {ticker:'TEAMG'},
  {ticker:'CMAN'},
  {ticker: 'SUPEREIF'},
  {ticker:'TFFIF'},
  {ticker:'AIMCG'},
  {ticker: 'SPRIME'},
  {ticker:'B-WORK'},
  {ticker:'BOFFICE'},
  {ticker: 'AIMIRT'},
  {ticker: 'IP'},
  {ticker: 'INSET'},
  {ticker: 'KUMWEL'},
  {ticker: 'ARIN'},
  {ticker: 'ACG'},
  {ticker: 'MITSIB'},
  {ticker: 'VL'},
  {ticker: 'ALL'},
  {ticker: 'GSC'},
  {ticker: 'CAZ'},
  {ticker: 'SAAM'},
  {ticker:'STI'},
  {ticker:'SISB'},
  {ticker:'CMC'},
  {ticker:'TIGER'},
  {ticker:'SONIC'},
  {ticker:'KWM'},
  {ticker:'TPLAS'},
  {ticker:'MVP'},
  {ticker:'DOD'},
  {ticker:'CHAYO'},
  {ticker:'ABM'},
  {ticker:'MM'},
  {ticker:'D'},
  {ticker:'SE'},
  {ticker:'HREIT'},
{ticker:'PTT'},
{ticker:'1DIV'},
{ticker:'2S'},
{ticker:'A'},
{ticker:'AAV'},
{ticker:'ABICO'},
{ticker:'ABPIF'},
{ticker:'ACAP'},
{ticker:'ACC'},
{ticker:'ADAM'},
{ticker:'ADB'},
{ticker:'ADVANC'},
{ticker:'AEC'},
{ticker:'AEONTS'},
{ticker:'AF'},
{ticker:'AFC'},
{ticker:'AGE'},
{ticker:'AH'},
{ticker:'AHC'},
{ticker:'AI'},
{ticker:'AIE'},
{ticker:'AIMIRT'},
{ticker:'AIRA'},
{ticker:'AIT'},
{ticker:'AJ'},
{ticker:'AJA'},
{ticker:'AKP'},
{ticker:'AKR'},
{ticker:'ALLA'},
{ticker:'ALT'},
{ticker:'ALUCON'},
{ticker:'AMA'},
{ticker:'AMANAH'},
{ticker:'AMARIN'},
{ticker:'AMATA'},
{ticker:'AMATAR'},
{ticker:'AMATAV'},
{ticker:'AMC'},
{ticker:'ANAN'},
{ticker:'AOT'},
{ticker:'AP'},
{ticker:'APCO'},
{ticker:'APCS'},
{ticker:'APEX'},
{ticker:'APURE'},
{ticker:'AQ'},
{ticker:'AQUA'},
{ticker:'ARIP'},
{ticker:'ARROW'},
{ticker:'AS'},
{ticker:'ASAP'},
{ticker:'ASEFA'},
{ticker:'ASIA'},
{ticker:'ASIAN'},
{ticker:'ASIMAR'},
{ticker:'ASK'},
{ticker:'ASN'},
{ticker:'ASP'},
{ticker:'ATP30'},
{ticker:'AU'},
{ticker:'AUCT'},
{ticker:'AYUD'},
{ticker:'B'},
{ticker:'BA'},
{ticker:'BAFS'},
{ticker:'BANPU'},
{ticker:'BAT-3K'},
{ticker:'BAY'},
{ticker:'BBL'},
{ticker:'BCH'},
{ticker:'BCP'},
{ticker:'BCPG'},
{ticker:'BDMS'},
{ticker:'BEAUTY'},
{ticker:'BEC'},
{ticker:'BEM'},
{ticker:'BFIT'},
{ticker:'BGRIM'},
{ticker:'BGT'},
{ticker:'BH'},
{ticker:'BIG'},
{ticker:'BIZ'},
{ticker:'BJC'},
{ticker:'BJCHI'},
{ticker:'BKD'},
{ticker:'BKI'},
{ticker:'BKKCP'},
{ticker:'BLA'},
{ticker:'BLAND'},
{ticker:'BLISS'},
{ticker:'BM'},
{ticker:'BMSCITH'},
{ticker:'BOFFICE'},
{ticker:'BOL'},
{ticker:'BPP'},
{ticker:'BR'},
{ticker:'BROCK'},
{ticker:'BROOK'},
{ticker:'BRR'},
{ticker:'BRRGIF'},
{ticker:'BSBM'},
{ticker:'BSET100'},
{ticker:'BSM'},
{ticker:'BTNC'},
{ticker:'BTS'},
{ticker:'BTSGIF'},
{ticker:'BTW'},
{ticker:'BUI'},
{ticker:'BWG'},
{ticker:'CBG'},
{ticker:'CCET'},
{ticker:'CCP'},
{ticker:'CEN'},
{ticker:'CENTEL'},
{ticker:'CFRESH'},
{ticker:'CGD'},
{ticker:'CGH'},
{ticker:'CHARAN'},
{ticker:'CHEWA'},
{ticker:'CHG'},
{ticker:'CHINA'},
{ticker:'CHO'},
{ticker:'CHOTI'},
{ticker:'CHOW'},
{ticker:'CHUO'},
{ticker:'CI'},
{ticker:'CIG'},
{ticker:'CIMBT'},
{ticker:'CITY'},
{ticker:'CK'},
{ticker:'CKP'},
{ticker:'CM'},
{ticker:'CMO'},
{ticker:'CMR'},
{ticker:'CNS'},
{ticker:'CNT'},
{ticker:'COL'},
{ticker:'COLOR'},
{ticker:'COM7'},
{ticker:'COMAN'},
{ticker:'CPALL'},
{ticker:'CPF'},
{ticker:'CPH'},
{ticker:'CPI'},
{ticker:'CPL'},
{ticker:'CPN'},
{ticker:'CPNCG'},
{ticker:'CPNREIT'},
{ticker:'CPR'},
{ticker:'CPT'},
{ticker:'CPTGF'},
{ticker:'CRANE'},
{ticker:'CRD'},
{ticker:'CRYSTAL'},
{ticker:'CSC'},
{ticker:'CSL'},
{ticker:'CSP'},
{ticker:'CSR'},
{ticker:'CSS'},
{ticker:'CTARAF'},
{ticker:'CTW'},
{ticker:'CWT'},
{ticker:'D'},
{ticker:'DCC'},
{ticker:'DCON'},
{ticker:'DCORP'},
{ticker:'DDD'},
{ticker:'DELTA'},
{ticker:'DEMCO'},
{ticker:'DIF'},
{ticker:'DIGI'},
{ticker:'DIMET'},
{ticker:'DNA'},
{ticker:'DREIT'},
{ticker:'DRT'},
{ticker:'DTAC'},
{ticker:'DTC'},
{ticker:'DTCI'},
{ticker:'EA'},
{ticker:'EARTH'},
{ticker:'EASON'},
{ticker:'EASTW'},
{ticker:'EBANK'},
{ticker:'ECF'},
{ticker:'ECL'},
{ticker:'ECOMM'},
{ticker:'EE'},
{ticker:'EFOOD'},
{ticker:'EFORL'},
{ticker:'EGATIF'},
{ticker:'EGCO'},
{ticker:'EIC'},
{ticker:'EICT'},
{ticker:'EKH'},
{ticker:'EMC'},
{ticker:'ENGY'},
{ticker:'ENY'},
{ticker:'EPCO'},
{ticker:'EPG'},
{ticker:'ERW'},
{ticker:'ERWPF'},
{ticker:'ESET50'},
{ticker:'ESSO'},
{ticker:'ESTAR'},
{ticker:'ETE'},
{ticker:'EVER'},
{ticker:'F&D'},
{ticker:'FANCY'},
{ticker:'FC'},
{ticker:'FE'},
{ticker:'FER'},
{ticker:'FLOYD'},
{ticker:'FMT'},
{ticker:'FN'},
{ticker:'FNS'},
{ticker:'FOCUS'},
{ticker:'FORTH'},
{ticker:'FPI'},
{ticker:'FPI-W1'},
{ticker:'FSMART'},
{ticker:'FSS'},
{ticker:'FTE'},
{ticker:'FUTUREPF'},
{ticker:'FVC'},
{ticker:'GAHREIT'},
{ticker:'GBX'},
{ticker:'GC'},
{ticker:'GCAP'},
{ticker:'GEL'},
{ticker:'GENCO'},
{ticker:'GFPT'},
{ticker:'GGC'},
{ticker:'GIFT'},
{ticker:'GJS'},
{ticker:'GL'},
{ticker:'GLAND'},
{ticker:'GLANDRT'},
{ticker:'GLD'},
{ticker:'GLOBAL'},
{ticker:'GLOW'},
{ticker:'GOLD'},
{ticker:'GOLDPF'},
{ticker:'GPI'},
{ticker:'GPSC'},
{ticker:'GRAMMY'},
{ticker:'GRAND'},
{ticker:'GREEN'},
{ticker:'GSTEL'},
{ticker:'GTB'},
{ticker:'GULF'},
{ticker:'GUNKUL'},
{ticker:'GVREIT'},
{ticker:'GYT'},
{ticker:'HANA'},
{ticker:'HARN'},
{ticker:'HFT'},
{ticker:'HMPRO'},
{ticker:'HOTPOT'},
{ticker:'HPF'},
{ticker:'HPT'},
{ticker:'HREIT'},
{ticker:'HTC'},
{ticker:'HTECH'},
{ticker:'HUMAN'},
{ticker:'HYDRO'},
{ticker:'ICC'},
{ticker:'ICHI'},
{ticker:'ICN'},
{ticker:'IEC'},
{ticker:'IFEC'},
{ticker:'IFS'},
{ticker:'IHL'},
{ticker:'III'},
{ticker:'ILINK'},
{ticker:'IMPACT'},
{ticker:'INET'},
{ticker:'INGRS'},
{ticker:'INOX'},
{ticker:'INSURE'},
{ticker:'INTUCH'},
{ticker:'IRC'},
{ticker:'IRCP'},
{ticker:'IRPC'},
{ticker:'IT'},
{ticker:'ITD'},
{ticker:'ITEL'},
{ticker:'IVL'},
{ticker:'J'},
{ticker:'JAS'},
{ticker:'JASIF'},
{ticker:'JCT'},
{ticker:'JKN'},
{ticker:'JMART'},
{ticker:'JMT'},
{ticker:'JSP'},
{ticker:'JTS'},
{ticker:'JUBILE'},
{ticker:'JUTHA'},
{ticker:'JWD'},
{ticker:'K'},
{ticker:'KAMART'},
{ticker:'KASET'},
{ticker:'KBANK'},
{ticker:'KBS'},
{ticker:'KC'},
{ticker:'KCAR'},
{ticker:'KCE'},
{ticker:'KCM'},
{ticker:'KDH'},
{ticker:'KGI'},
{ticker:'KIAT'},
{ticker:'KKC'},
{ticker:'KKP'},
{ticker:'KOOL'},
{ticker:'KPNPF'},
{ticker:'KSL'},
{ticker:'KTB'},
{ticker:'KTC'},
{ticker:'KTIS'},
{ticker:'KWC'},
{ticker:'KWG'},
{ticker:'KYE'},
{ticker:'L&E'},
{ticker:'LALIN'},
{ticker:'LANNA'},
{ticker:'LDC'},
{ticker:'LEE'},
{ticker:'LH'},
{ticker:'LHFG'},
{ticker:'LHHOTEL'},
{ticker:'LHK'},
{ticker:'LHPF'},
{ticker:'LHSC'},
{ticker:'LIT'},
{ticker:'LOXLEY'},
{ticker:'LPH'},
{ticker:'LPN'},
{ticker:'LRH'},
{ticker:'LST'},
{ticker:'LTX'},
{ticker:'LUXF'},
{ticker:'LVT'},
{ticker:'M'},
{ticker:'M-CHAI'},
{ticker:'M-II'},
{ticker:'M-PAT'},
{ticker:'M-STOR'},
{ticker:'MACO'},
{ticker:'MAJOR'},
{ticker:'MAKRO'},
{ticker:'MALEE'},
{ticker:'MANRIN'},
{ticker:'MATCH'},
{ticker:'MATI'},
{ticker:'MAX'},
{ticker:'MBAX'},
{ticker:'MBK'},
{ticker:'MBKET'},
{ticker:'MC'},
{ticker:'MCOT'},
{ticker:'MCS'},
{ticker:'MDX'},
{ticker:'MEGA'},
{ticker:'METCO'},
{ticker:'MFC'},
{ticker:'MFEC'},
{ticker:'MGE'},
{ticker:'MGT'},
{ticker:'MIDA'},
{ticker:'MILL'},
{ticker:'MINT'},
{ticker:'MIPF'},
{ticker:'MIT'},
{ticker:'MJD'},
{ticker:'MJLF'},
{ticker:'MK'},
{ticker:'ML'},
{ticker:'MM'},
{ticker:'MNIT'},
{ticker:'MNIT2'},
{ticker:'MNRF'},
{ticker:'MODERN'},
{ticker:'MONO'},
{ticker:'MONTRI'},
{ticker:'MOONG'},
{ticker:'MPG'},
{ticker:'MPIC'},
{ticker:'MSC'},
{ticker:'MTI'},
{ticker:'MTLS'},
{ticker:'NBC'},
{ticker:'NC'},
{ticker:'NCH'},
{ticker:'NCL'},
{ticker:'NDR'},
{ticker:'NEP'},
{ticker:'NETBAY'},
{ticker:'NEW'},
{ticker:'NEWS'},
{ticker:'NINE'},
{ticker:'NKI'},
{ticker:'NMG'},
{ticker:'NNCL'},
{ticker:'NOBLE'},
{ticker:'NOK'},
{ticker:'NPK'},
{ticker:'NPP'},
{ticker:'NSI'},
{ticker:'NTV'},
{ticker:'NUSA'},
{ticker:'NVD'},
{ticker:'NWR'},
{ticker:'NYT'},
{ticker:'OCC'},
{ticker:'OCEAN'},
{ticker:'OGC'},
{ticker:'OHTL'},
{ticker:'OISHI'},
{ticker:'ORI'},
{ticker:'OTO'},
{ticker:'PACE'},
{ticker:'PAE'},
{ticker:'PAF'},
{ticker:'PAP'},
{ticker:'PATO'},
{ticker:'PB'},
{ticker:'PCSGH'},
{ticker:'PDG'},
{ticker:'PDI'},
{ticker:'PDJ'},
{ticker:'PE'},
{ticker:'PERM'},
{ticker:'PF'},
{ticker:'PG'},
{ticker:'PHOL'},
{ticker:'PICO'},
{ticker:'PIMO'},
{ticker:'PJW'},
{ticker:'PK'},
{ticker:'PL'},
{ticker:'PLANB'},
{ticker:'PLANET'},
{ticker:'PLAT'},
{ticker:'PLE'},
{ticker:'PM'},
{ticker:'PMTA'},
{ticker:'POLAR'},
{ticker:'POPF'},
{ticker:'PORT'},
{ticker:'POST'},
{ticker:'PPF'},
{ticker:'PPM'},
{ticker:'PPP'},
{ticker:'PPS'},
{ticker:'PRAKIT'},
{ticker:'PREB'},
{ticker:'PRECHA'},
{ticker:'PRG'},
{ticker:'PRIN'},
{ticker:'PRINC'},
{ticker:'PRM'},
{ticker:'PRO'},
{ticker:'PSH'},
{ticker:'PSL'},
{ticker:'PSTC'},
{ticker:'PT'},
{ticker:'PTG'},
{ticker:'PTL'},
{ticker:'PTT'},
{ticker:'PTTEP'},
{ticker:'PTTGC'},
{ticker:'PYLON'},
{ticker:'Q-CON'},
{ticker:'QH'},
{ticker:'QHHR'},
{ticker:'QHOP'},
{ticker:'QHPF'},
{ticker:'QLT'},
{ticker:'QTC'},
{ticker:'RAM'},
{ticker:'RATCH'},
{ticker:'RCI'},
{ticker:'RCL'},
{ticker:'RICH'},
{ticker:'RICHY'},
{ticker:'RJH'},
{ticker:'RML'},
{ticker:'ROBINS'},
{ticker:'ROCK'},
{ticker:'ROH'},
{ticker:'ROJNA'},
{ticker:'RP'},
{ticker:'RPC'},
{ticker:'RPH'},
{ticker:'RS'},
{ticker:'RSP'},
{ticker:'RWI'},
{ticker:'S'},
{ticker:'S&J'},
{ticker:'S11'},
{ticker:'SABINA'},
{ticker:'SALEE'},
{ticker:'SAM'},
{ticker:'SAMART'},
{ticker:'SAMCO'},
{ticker:'SAMTEL'},
{ticker:'SANKO'},
{ticker:'SAPPE'},
{ticker:'SAT'},
{ticker:'SAUCE'},
{ticker:'SAWAD'},
{ticker:'SAWANG'},
{ticker:'SBPF'},
{ticker:'SC'},
{ticker:'SCB'},
{ticker:'SCC'},
{ticker:'SCCC'},
{ticker:'SCG'},
{ticker:'SCI'},
{ticker:'SCN'},
{ticker:'SCP'},
{ticker:'SDC'},
{ticker:'SE'},
{ticker:'SE-ED'},
{ticker:'SEAFCO'},
{ticker:'SEAOIL'},
{ticker:'SELIC'},
{ticker:'SENA'},
{ticker:'SF'},
{ticker:'SFP'},
{ticker:'SGF'},
{ticker:'SGP'},
{ticker:'SHANG'},
{ticker:'SHREIT'},
{ticker:'SIAM'},
{ticker:'SIMAT'},
{ticker:'SINGER'},
{ticker:'SIRI'},
{ticker:'SIRIP'},
{ticker:'SIS'},
{ticker:'SITHAI'},
{ticker:'SKE'},
{ticker:'SKN'},
{ticker:'SKR'},
{ticker:'SKY'},
{ticker:'SLP'},
{ticker:'SMART'},
{ticker:'SMIT'},
{ticker:'SMK'},
{ticker:'SMM'},
{ticker:'SMPC'},
{ticker:'SMT'},
{ticker:'SNC'},
{ticker:'SNP'},
{ticker:'SOLAR'},
{ticker:'SORKON'},
{ticker:'SPA'},
{ticker:'SPACK'},
{ticker:'SPALI'},
{ticker:'SPC'},
{ticker:'SPCG'},
{ticker:'SPF'},
{ticker:'SPG'},
{ticker:'SPI'},
{ticker:'SPORT'},
{ticker:'SPPT'},
{ticker:'SPRC'},
{ticker:'SPVI'},
{ticker:'SQ'},
{ticker:'SR'},
{ticker:'SRICHA'},
{ticker:'SRIPANWA'},
{ticker:'SSC'},
{ticker:'SSF'},
{ticker:'SSI'},
{ticker:'SSP'},
{ticker:'SSPF'},
{ticker:'SSSC'},
{ticker:'SST'},
{ticker:'SSTPF'},
{ticker:'SSTRT'},
{ticker:'STA'},
{ticker:'STANLY'},
{ticker:'STAR'},
{ticker:'STEC'},
{ticker:'STHAI'},
{ticker:'STPI'},
{ticker:'SUC'},
{ticker:'SUN'},
{ticker:'SUPER'},
{ticker:'SUSCO'},
{ticker:'SUTHA'},
{ticker:'SVH'},
{ticker:'SVI'},
{ticker:'SVOA'},
{ticker:'SWC'},
{ticker:'SYMC'},
{ticker:'SYNEX'},
{ticker:'SYNTEC'},
{ticker:'T'},
{ticker:'TACC'},
{ticker:'TAE'},
{ticker:'TAKUNI'},
{ticker:'TAPAC'},
{ticker:'TASCO'},
{ticker:'TBSP'},
{ticker:'TC'},
{ticker:'TCAP'},
{ticker:'TCB'},
{ticker:'TCC'},
{ticker:'TCCC'},
{ticker:'TCJ'},
{ticker:'TCMC'},
{ticker:'TCOAT'},
{ticker:'TDEX'},
{ticker:'TEAM'},
{ticker:'TFD'},
{ticker:'TFG'},
{ticker:'TFI'},
{ticker:'TFMAMA'},
{ticker:'TGCI'},
{ticker:'TGOLDETF'},
{ticker:'TGPRO'},
{ticker:'TH'},
{ticker:'TH100'},
{ticker:'THAI'},
{ticker:'THANA'},
{ticker:'THANI'},
{ticker:'THCOM'},
{ticker:'THE'},
{ticker:'THG'},
{ticker:'THIP'},
{ticker:'THL'},
{ticker:'THMUI'},
{ticker:'THRE'},
{ticker:'THREL'},
{ticker:'TIC'},
{ticker:'TICON'},
{ticker:'TIF1'},
{ticker:'TIP'},
{ticker:'TIPCO'},
{ticker:'TISCO'},
{ticker:'TITLE'},
{ticker:'TIW'},
{ticker:'TK'},
{ticker:'TKN'},
{ticker:'TKS'},
{ticker:'TKT'},
{ticker:'TLGF'},
{ticker:'TLHPF'},
{ticker:'TLUXE'},
{ticker:'TM'},
{ticker:'TMB'},
{ticker:'TMC'},
{ticker:'TMD'},
{ticker:'TMI'},
{ticker:'TMILL'},
{ticker:'TMT'},
{ticker:'TMW'},
{ticker:'TNDT'},
{ticker:'TNH'},
{ticker:'TNITY'},
{ticker:'TNL'},
{ticker:'TNP'},
{ticker:'TNPC'},
{ticker:'TNPF'},
{ticker:'TNR'},
{ticker:'TOA'},
{ticker:'TOG'},
{ticker:'TOP'},
{ticker:'TOPP'},
{ticker:'TPA'},
{ticker:'TPAC'},
{ticker:'TPBI'},
{ticker:'TPCH'},
{ticker:'TPCORP'},
{ticker:'TPIPL'},
{ticker:'TPIPP'},
{ticker:'TPOLY'},
{ticker:'TPP'},
{ticker:'TPRIME'},
{ticker:'TR'},
{ticker:'TRC'},
{ticker:'TREIT'},
{ticker:'TRITN'},
{ticker:'TRT'},
{ticker:'TRU'},
{ticker:'TRUBB'},
{ticker:'TRUE'},
{ticker:'TSC'},
{ticker:'TSE'},
{ticker:'TSF'},
{ticker:'TSI'},
{ticker:'TSR'},
{ticker:'TSTE'},
{ticker:'TSTH'},
{ticker:'TTA'},
{ticker:'TTCL'},
{ticker:'TTI'},
{ticker:'TTL'},
{ticker:'TTLPF'},
{ticker:'TTTM'},
{ticker:'TTW'},
{ticker:'TU'},
{ticker:'TU-PF'},
{ticker:'TUCC'},
{ticker:'TVD'},
{ticker:'TVI'},
{ticker:'TVO'},
{ticker:'TVT'},
{ticker:'TWP'},
{ticker:'TWPC'},
{ticker:'TWZ'},
{ticker:'TYCN'},
{ticker:'U'},
{ticker:'UAC'},
{ticker:'UBIS'},
{ticker:'UEC'},
{ticker:'UKEM'},
{ticker:'UMI'},
{ticker:'UMS'},
{ticker:'UNIQ'},
{ticker:'UOBKH'},
{ticker:'UP'},
{ticker:'UPA'},
{ticker:'UPF'},
{ticker:'UPOIC'},
{ticker:'URBNPF'},
{ticker:'UREKA'},
{ticker:'UT'},
{ticker:'UTP'},
{ticker:'UV'},
{ticker:'UVAN'},
{ticker:'UWC'},
{ticker:'VARO'},
{ticker:'VCOM'},
{ticker:'VGI'},
{ticker:'VIBHA'},
{ticker:'VIH'},
{ticker:'VNG'},
{ticker:'VNT'},
{ticker:'VPO'},
{ticker:'VTE'},
{ticker:'WACOAL'},
{ticker:'WAVE'},
{ticker:'WG'},
{ticker:'WHA'},
{ticker:'WHABT'},
{ticker:'WHART'},
{ticker:'WHAUP'},
{ticker:'WICE'},
{ticker:'WIIK'},
{ticker:'WIN'},
{ticker:'WINNER'},
{ticker:'WORK'},
{ticker:'WPH'},
{ticker:'XO'},
{ticker:'YCI'},
{ticker:'YNP'},
{ticker:'YUASA'},
{ticker:'ZIGA'},
{ticker:'ZMICO'}]

with open('output', 'wb', newline = '' ) as my file:
  wr = csv.writer(myfile, quoting = csv.QUOTE_ALL)
  wr.writerow(contexts)
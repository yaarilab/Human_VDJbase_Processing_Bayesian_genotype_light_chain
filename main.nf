$HOSTNAME = ""
params.outdir = 'results'  

//* params.nproc =  10  //* @input @description:"number of processes cores to use"
//* params.chain =  "IGL"  //* @input @description:"chain"

// Process Parameters for First_Alignment_IgBlastn:
params.First_Alignment_IgBlastn.num_threads = params.nproc
params.First_Alignment_IgBlastn.ig_seqtype = "Ig"
params.First_Alignment_IgBlastn.outfmt = "MakeDb"
params.First_Alignment_IgBlastn.num_alignments_V = "10"
params.First_Alignment_IgBlastn.domain_system = "imgt"


params.First_Alignment_MakeDb.failed = "true"
params.First_Alignment_MakeDb.format = "airr"
params.First_Alignment_MakeDb.regions = "default"
params.First_Alignment_MakeDb.extended = "true"
params.First_Alignment_MakeDb.asisid = "false"
params.First_Alignment_MakeDb.asiscalls = "false"
params.First_Alignment_MakeDb.inferjunction = "false"
params.First_Alignment_MakeDb.partial = "false"
params.First_Alignment_MakeDb.name_alignment = "First_Alignment"

// Process Parameters for First_Alignment_Collapse_AIRRseq:
params.First_Alignment_Collapse_AIRRseq.conscount_min = 2
params.First_Alignment_Collapse_AIRRseq.n_max = 10
params.First_Alignment_Collapse_AIRRseq.name_alignment = "First_Alignment"

// Process Parameters for ogrdbstats_report_first_alignment:
params.ogrdbstats_report_first_alignment.chain = params.chain+"V"

// Process Parameters for Undocumented_Alleles:
params.Undocumented_Alleles.chain = "IGH"
params.Undocumented_Alleles.num_threads = params.nproc
params.Undocumented_Alleles.germline_min = 200
params.Undocumented_Alleles.min_seqs = 50
params.Undocumented_Alleles.auto_mutrange = "true"
params.Undocumented_Alleles.mut_range = "1:10"
params.Undocumented_Alleles.pos_range = "1:318"
params.Undocumented_Alleles.y_intercept = 0.125
params.Undocumented_Alleles.alpha = 0.05
params.Undocumented_Alleles.j_max = 0.15
params.Undocumented_Alleles.min_frac = 0.75


// part 3
// Process Parameters for Second_Alignment_IgBlastn:
params.Second_Alignment_IgBlastn.num_threads = params.nproc
params.Second_Alignment_IgBlastn.ig_seqtype = "Ig"
params.Second_Alignment_IgBlastn.outfmt = "MakeDb"
params.Second_Alignment_IgBlastn.num_alignments_V = "10"
params.Second_Alignment_IgBlastn.domain_system = "imgt"

params.Second_Alignment_MakeDb.failed = "true"
params.Second_Alignment_MakeDb.format = "airr"
params.Second_Alignment_MakeDb.regions = "default"
params.Second_Alignment_MakeDb.extended = "true"
params.Second_Alignment_MakeDb.asisid = "false"
params.Second_Alignment_MakeDb.asiscalls = "false"
params.Second_Alignment_MakeDb.inferjunction = "false"
params.Second_Alignment_MakeDb.partial = "false"
params.Second_Alignment_MakeDb.name_alignment = "Second_Alignment"

// part 5

// Process Parameters for TIgGER_bayesian_genotype_Inference:
params.TIgGER_bayesian_genotype_Inference_v_call.call = "v_call"
params.TIgGER_bayesian_genotype_Inference_v_call.seq = "sequence_alignment"
params.TIgGER_bayesian_genotype_Inference_v_call.find_unmutated = "false"
params.TIgGER_bayesian_genotype_Inference_v_call.single_assignments = "false"


// Process Parameters for TIgGER_bayesian_genotype_Inference_d_call:
params.TIgGER_bayesian_genotype_Inference_d_call.call = "d_call"
params.TIgGER_bayesian_genotype_Inference_d_call.seq = "sequence_alignment"
params.TIgGER_bayesian_genotype_Inference_d_call.find_unmutated = "false"
params.TIgGER_bayesian_genotype_Inference_d_call.single_assignments = "true"
params.TIgGER_bayesian_genotype_Inference_d_call.chain = params.chain

// Process Parameters for TIgGER_bayesian_genotype_Inference_j_call:
params.TIgGER_bayesian_genotype_Inference_j_call.call = "j_call"
params.TIgGER_bayesian_genotype_Inference_j_call.seq = "sequence_alignment"
params.TIgGER_bayesian_genotype_Inference_j_call.find_unmutated = "false"
params.TIgGER_bayesian_genotype_Inference_j_call.single_assignments = "true"


// part 6

// Process Parameters for Third_Alignment_IgBlastn:
params.Third_Alignment_IgBlastn.num_threads = params.nproc
params.Third_Alignment_IgBlastn.ig_seqtype = "Ig"
params.Third_Alignment_IgBlastn.outfmt = "MakeDb"
params.Third_Alignment_IgBlastn.num_alignments_V = "10"
params.Third_Alignment_IgBlastn.domain_system = "imgt"

params.Third_Alignment_MakeDb.failed = "true"
params.Third_Alignment_MakeDb.format = "airr"
params.Third_Alignment_MakeDb.regions = "default"
params.Third_Alignment_MakeDb.extended = "true"
params.Third_Alignment_MakeDb.asisid = "false"
params.Third_Alignment_MakeDb.asiscalls = "false"
params.Third_Alignment_MakeDb.inferjunction = "false"
params.Third_Alignment_MakeDb.partial = "false"
params.Third_Alignment_MakeDb.name_alignment = "Finale"

// part 7

// Process Parameters for ogrdbstats_report:
params.ogrdbstats_report.chain = params.chain + "V"


if (!params.v_germline_file){params.v_germline_file = ""} 
if (!params.d_germline){params.d_germline = ""} 
if (!params.j_germline){params.j_germline = ""} 
if (!params.airr_seq){params.airr_seq = ""} 
// Stage empty file to be used as an optional input where required
ch_empty_file_1 = file("$baseDir/.emptyfiles/NO_FILE_1", hidden:true)

Channel.fromPath(params.v_germline_file, type: 'any').map{ file -> tuple(file.baseName, file) }.into{g_2_germlineFastaFile_g_8;g_2_germlineFastaFile_g_68;g_2_germlineFastaFile_g0_22;g_2_germlineFastaFile_g0_12}
Channel.fromPath(params.d_germline, type: 'any').map{ file -> tuple(file.baseName, file) }.into{g_3_germlineFastaFile_g_75;g_3_germlineFastaFile_g0_16;g_3_germlineFastaFile_g0_12;g_3_germlineFastaFile_g11_16;g_3_germlineFastaFile_g11_12}
Channel.fromPath(params.j_germline, type: 'any').map{ file -> tuple(file.baseName, file) }.into{g_4_germlineFastaFile_g_31;g_4_germlineFastaFile_g0_17;g_4_germlineFastaFile_g0_12;g_4_germlineFastaFile_g11_17;g_4_germlineFastaFile_g11_12}
Channel.fromPath(params.airr_seq, type: 'any').map{ file -> tuple(file.baseName, file) }.into{g_44_fastaFile_g_73;g_44_fastaFile_g0_12;g_44_fastaFile_g0_9}


process First_Alignment_D_MakeBlastDb {

input:
 set val(db_name), file(germlineFile) from g_3_germlineFastaFile_g0_16

output:
 file "${db_name}"  into g0_16_germlineDb0_g0_9

script:

if(germlineFile.getName().endsWith("fasta")){
	"""
	sed -e '/^>/! s/[.]//g' ${germlineFile} > tmp_germline.fasta
	mkdir -m777 ${db_name}
	makeblastdb -parse_seqids -dbtype nucl -in tmp_germline.fasta -out ${db_name}/${db_name}
	"""
}else{
	"""
	echo something if off
	"""
}

}


process First_Alignment_J_MakeBlastDb {

input:
 set val(db_name), file(germlineFile) from g_4_germlineFastaFile_g0_17

output:
 file "${db_name}"  into g0_17_germlineDb0_g0_9

script:

if(germlineFile.getName().endsWith("fasta")){
	"""
	sed -e '/^>/! s/[.]//g' ${germlineFile} > tmp_germline.fasta
	mkdir -m777 ${db_name}
	makeblastdb -parse_seqids -dbtype nucl -in tmp_germline.fasta -out ${db_name}/${db_name}
	"""
}else{
	"""
	echo something if off
	"""
}

}


process First_Alignment_V_MakeBlastDb {

input:
 set val(db_name), file(germlineFile) from g_2_germlineFastaFile_g0_22

output:
 file "${db_name}"  into g0_22_germlineDb0_g0_9

script:

if(germlineFile.getName().endsWith("fasta")){
	"""
	sed -e '/^>/! s/[.]//g' ${germlineFile} > tmp_germline.fasta
	mkdir -m777 ${db_name}
	makeblastdb -parse_seqids -dbtype nucl -in tmp_germline.fasta -out ${db_name}/${db_name}
	"""
}else{
	"""
	echo something if off
	"""
}

}


process First_Alignment_IgBlastn {

input:
 set val(name),file(fastaFile) from g_44_fastaFile_g0_9
 file db_v from g0_22_germlineDb0_g0_9
 file db_d from g0_16_germlineDb0_g0_9
 file db_j from g0_17_germlineDb0_g0_9

output:
 set val(name), file("${outfile}") optional true  into g0_9_igblastOut0_g0_12

script:
num_threads = params.First_Alignment_IgBlastn.num_threads
ig_seqtype = params.First_Alignment_IgBlastn.ig_seqtype
outfmt = params.First_Alignment_IgBlastn.outfmt
num_alignments_V = params.First_Alignment_IgBlastn.num_alignments_V
domain_system = params.First_Alignment_IgBlastn.domain_system
auxiliary_data = params.First_Alignment_IgBlastn.auxiliary_data

randomString = org.apache.commons.lang.RandomStringUtils.random(9, true, true)
outname = name + "_" + randomString
outfile = (outfmt=="MakeDb") ? name+"_"+randomString+".out" : name+"_"+randomString+".tsv"
outfmt = (outfmt=="MakeDb") ? "'7 std qseq sseq btop'" : "19"

if(db_v.toString()!="" && db_d.toString()!="" && db_j.toString()!=""){
	"""
	igblastn -query ${fastaFile} \
		-germline_db_V ${db_v}/${db_v} \
		-germline_db_D ${db_d}/${db_d} \
		-germline_db_J ${db_j}/${db_j} \
		-num_alignments_V ${num_alignments_V} \
		-domain_system imgt \
		-auxiliary_data ${auxiliary_data} \
		-outfmt ${outfmt} \
		-num_threads ${num_threads} \
		-out ${outfile}
	"""
}else{
	"""
	"""
}

}


process First_Alignment_MakeDb {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*_db-pass.tsv$/) "first_rearrangement/$filename"}
publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*_db-fail.tsv$/) "first_rearrangement/$filename"}
input:
 set val(name),file(fastaFile) from g_44_fastaFile_g0_12
 set val(name_igblast),file(igblastOut) from g0_9_igblastOut0_g0_12
 set val(name1), file(v_germline_file) from g_2_germlineFastaFile_g0_12
 set val(name2), file(d_germline_file) from g_3_germlineFastaFile_g0_12
 set val(name3), file(j_germline_file) from g_4_germlineFastaFile_g0_12

output:
 set val(name_igblast),file("*_db-pass.tsv") optional true  into g0_12_outputFileTSV0_g0_19
 set val("reference_set"), file("${reference_set}") optional true  into g0_12_germlineFastaFile1_g_68
 set val(name_igblast),file("*_db-fail.tsv") optional true  into g0_12_outputFileTSV22

script:

failed = params.First_Alignment_MakeDb.failed
format = params.First_Alignment_MakeDb.format
regions = params.First_Alignment_MakeDb.regions
extended = params.First_Alignment_MakeDb.extended
asisid = params.First_Alignment_MakeDb.asisid
asiscalls = params.First_Alignment_MakeDb.asiscalls
inferjunction = params.First_Alignment_MakeDb.inferjunction
partial = params.First_Alignment_MakeDb.partial
name_alignment = params.First_Alignment_MakeDb.name_alignment

failed = (failed=="true") ? "--failed" : ""
format = (format=="changeo") ? "--format changeo" : ""
extended = (extended=="true") ? "--extended" : ""
regions = (regions=="rhesus-igl") ? "--regions rhesus-igl" : ""
asisid = (asisid=="true") ? "--asis-id" : ""
asiscalls = (asiscalls=="true") ? "--asis-calls" : ""
inferjunction = (inferjunction=="true") ? "--infer-junction" : ""
partial = (partial=="true") ? "--partial" : ""

reference_set = "reference_set_makedb_"+name_alignment+".fasta"

outname = name_igblast+'_'+name_alignment

if(igblastOut.getName().endsWith(".out")){
	"""
	
	cat ${v_germline_file} ${d_germline_file} ${j_germline_file} > ${reference_set}
	
	MakeDb.py igblast \
		-s ${fastaFile} \
		-i ${igblastOut} \
		-r ${v_germline_file} ${d_germline_file} ${j_germline_file} \
		--log MD_${name}.log \
		--outname ${outname}\
		${extended} \
		${failed} \
		${format} \
		${regions} \
		${asisid} \
		${asiscalls} \
		${inferjunction} \
		${partial}
	"""
}else{
	"""
	
	"""
}

}


process First_Alignment_Collapse_AIRRseq {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /${outfile}+passed.tsv$/) "first_rearrangement/$filename"}
publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /${outfile}+failed.*$/) "first_rearrangement/$filename"}
input:
 set val(name),file(airrFile) from g0_12_outputFileTSV0_g0_19

output:
 set val(name), file("${outfile}"+"passed.tsv") optional true  into g0_19_outputFileTSV0_g_68, g0_19_outputFileTSV0_g_8, g0_19_outputFileTSV0_g_80
 set val(name), file("${outfile}"+"failed*") optional true  into g0_19_outputFileTSV11

script:
conscount_min = params.First_Alignment_Collapse_AIRRseq.conscount_min
n_max = params.First_Alignment_Collapse_AIRRseq.n_max
name_alignment = params.First_Alignment_Collapse_AIRRseq.name_alignment


outfile = airrFile.toString() - '.tsv' + name_alignment + "_collapsed-"

if(airrFile.getName().endsWith(".tsv")){	
	"""
	#!/usr/bin/env python3
	
	from pprint import pprint
	from collections import OrderedDict,Counter
	import itertools as it
	import datetime
	import pandas as pd
	import glob, os
	import numpy as np
	import re
	
	# column types default
	
	# dtype_default={'junction_length': 'Int64', 'np1_length': 'Int64', 'np2_length': 'Int64', 'v_sequence_start': 'Int64', 'v_sequence_end': 'Int64', 'v_germline_start': 'Int64', 'v_germline_end': 'Int64', 'd_sequence_start': 'Int64', 'd_sequence_end': 'Int64', 'd_germline_start': 'Int64', 'd_germline_end': 'Int64', 'j_sequence_start': 'Int64', 'j_sequence_end': 'Int64', 'j_germline_start': 'Int64', 'j_germline_end': 'Int64', 'v_score': 'Int64', 'v_identity': 'Int64', 'v_support': 'Int64', 'd_score': 'Int64', 'd_identity': 'Int64', 'd_support': 'Int64', 'j_score': 'Int64', 'j_identity': 'Int64', 'j_support': 'Int64'}
	
	SPLITSIZE=2
	
	class CollapseDict:
	    def __init__(self,iterable=(),_depth=0,
	                 nlim=10,conscount_flag=False):
	        self.lowqual={}
	        self.seqs = {}
	        self.children = {}
	        self.depth=_depth
	        self.nlim=nlim
	        self.conscount=conscount_flag
	        for fseq in iterable:
	            self.add(fseq)
	
	    def split(self):
	        newseqs = {}
	        for seq in self.seqs:
	            if len(seq)==self.depth:
	                newseqs[seq]=self.seqs[seq]
	            else:
	                if seq[self.depth] not in self.children:
	                    self.children[seq[self.depth]] = CollapseDict(_depth=self.depth+1)
	                self.children[seq[self.depth]].add(self.seqs[seq],seq)
	        self.seqs=newseqs
	
	    def add(self,fseq,key=None):
	        #if 'duplicate_count' not in fseq: fseq['duplicate_count']='1'
	        if 'KEY' not in fseq:
	            fseq['KEY']=fseq['sequence_vdj'].replace('-','').replace('.','')
	        if 'ISOTYPECOUNTER' not in fseq:
	            fseq['ISOTYPECOUNTER']=Counter([fseq['c_call']])
	        if 'VGENECOUNTER' not in fseq:
	            fseq['VGENECOUNTER']=Counter([fseq['v_call']])
	        if 'JGENECOUNTER' not in fseq:
	            fseq['JGENECOUNTER']=Counter([fseq['j_call']])
	        if key is None:
	            key=fseq['KEY']
	        if self.depth==0:
	            if (not fseq['j_call'] or not fseq['v_call']):
	                return
	            if fseq['sequence_vdj'].count('N')>self.nlim:
	                if key in self.lowqual:
	                    self.lowqual[key] = combine(self.lowqual[key],fseq,self.conscount)
	                else:
	                    self.lowqual[key] = fseq
	                return
	        if len(self.seqs)>SPLITSIZE:
	            self.split()
	        if key in self.seqs:
	            self.seqs[key] = combine(self.seqs[key],fseq,self.conscount)
	        elif (self.children is not None and
	              len(key)>self.depth and
	              key[self.depth] in self.children):
	            self.children[key[self.depth]].add(fseq,key)
	        else:
	            self.seqs[key] = fseq
	
	    def __iter__(self):
	        yield from self.seqs.items()
	        for d in self.children.values():
	            yield from d
	        yield from self.lowqual.items()
	
	    def neighbors(self,seq):
	        def nfil(x): return similar(seq,x)
	        yield from filter(nfil,self.seqs)
	        if len(seq)>self.depth:
	            for d in [self.children[c]
	                      for c in self.children
	                      if c=='N' or seq[self.depth]=='N' or c==seq[self.depth]]:
	                yield from d.neighbors(seq)
	
	    def fixedseqs(self):
	        return self
	        ncd = CollapseDict()
	        for seq,fseq in self:
	            newseq=seq
	            if 'N' in seq:
	                newseq=consensus(seq,self.neighbors(seq))
	                fseq['KEY']=newseq
	            ncd.add(fseq,newseq)
	        return ncd
	
	
	    def __len__(self):
	        return len(self.seqs)+sum(len(c) for c in self.children.values())+len(self.lowqual)
	
	def combine(f1,f2, conscount_flag):
	    def val(f): return -f['KEY'].count('N'),(int(f['consensus_count']) if 'consensus_count' in f else 0)
	    targ = (f1 if val(f1) >= val(f2) else f2).copy()
	    if conscount_flag:
	        targ['consensus_count'] =  int(f1['consensus_count'])+int(f2['consensus_count'])
	    targ['duplicate_count'] =  int(f1['duplicate_count'])+int(f2['duplicate_count'])
	    targ['ISOTYPECOUNTER'] = f1['ISOTYPECOUNTER']+f2['ISOTYPECOUNTER']
	    targ['VGENECOUNTER'] = f1['VGENECOUNTER']+f2['VGENECOUNTER']
	    targ['JGENECOUNTER'] = f1['JGENECOUNTER']+f2['JGENECOUNTER']
	    return targ
	
	def similar(s1,s2):
	    return len(s1)==len(s2) and all((n1==n2 or n1=='N' or n2=='N')
	                                  for n1,n2 in zip(s1,s2))
	
	def basecon(bases):
	    bases.discard('N')
	    if len(bases)==1: return bases.pop()
	    else: return 'N'
	
	def consensus(seq,A):
	    return ''.join((basecon(set(B)) if s=='N' else s) for (s,B) in zip(seq,zip(*A)))
	
	n_lim = int('${n_max}')
	conscount_filter = int('${conscount_min}')
	
	df = pd.read_csv('${airrFile}', sep = '\t') #, dtype = dtype_default)
	
	# make sure that all columns are int64 for createGermline
	idx_col = df.columns.get_loc("cdr3")
	cols =  [col for col in df.iloc[:,0:idx_col].select_dtypes('float64').columns.values.tolist() if not re.search('support|score|identity|freq', col)]
	df[cols] = df[cols].apply(lambda x: pd.to_numeric(x.replace("<NA>",np.NaN), errors = "coerce").astype("Int64"))
	
	conscount_flag = False
	if 'consensus_count' in df: conscount_flag = True
	if not 'duplicate_count' in df:
	    df['duplicate_count'] = 1
	if not 'c_call' in df or not 'isotype' in df or not 'prcons' in df or not 'primer' in df or not 'reverse_primer' in df:
	    if 'c_call' in df:
	        df['c_call'] = df['c_call']
	    elif 'isotype' in df:
	        df['c_call'] = df['isotype']
	    elif 'primer' in df:
	        df['c_call'] = df['primer']
	    elif 'reverse_primer' in df:
	        df['c_call'] = df['reverse_primer']    
	    elif 'prcons' in df:
	        df['c_call'] = df['prcons']
	    elif 'barcode' in df:
	        df['c_call'] = df['barcode']
	    else:
	        df['c_call'] = 'Ig'
	
	# removing sequenes with duplicated sequence id    
	dup_n = df[df.columns[0]].count()
	df = df.drop_duplicates(subset='sequence_id', keep='first')
	dup_n = str(dup_n - df[df.columns[0]].count())
	df['c_call'] = df['c_call'].astype('str').replace('<NA>','Ig')
	#df['consensus_count'].fillna(2, inplace=True)
	nrow_i = df[df.columns[0]].count()
	df = df[df.apply(lambda x: x['sequence_alignment'][0:(x['v_germline_end']-1)].count('N')<=n_lim, axis = 1)]
	low_n = str(nrow_i-df[df.columns[0]].count())
	
	df['sequence_vdj'] = df.apply(lambda x: x['sequence_alignment'].replace('-','').replace('.',''), axis = 1)
	header=list(df.columns)
	fasta_ = df.to_dict(orient='records')
	c = CollapseDict(fasta_,nlim=10)
	d=c.fixedseqs()
	header.append('ISOTYPECOUNTER')
	header.append('VGENECOUNTER')
	header.append('JGENECOUNTER')
	
	rec_list = []
	for i, f in enumerate(d):
	    rec=f[1]
	    rec['sequence']=rec['KEY']
	    rec['consensus_count']=int(rec['consensus_count']) if conscount_flag else None
	    rec['duplicate_count']=int(rec['duplicate_count'])
	    rec_list.append(rec)
	df2 = pd.DataFrame(rec_list, columns = header)        
	
	df2 = df2.drop('sequence_vdj', axis=1)
	
	collapse_n = str(df[df.columns[0]].count()-df2[df2.columns[0]].count())

	# removing sequences without J assignment and non functional
	nrow_i = df2[df2.columns[0]].count()
	cond = (~df2['j_call'].str.contains('J')|df2['productive'].isin(['F','FALSE','False']))
	df_non = df2[cond]
	
	
	df2 = df2[df2['productive'].isin(['T','TRUE','True'])]
	cond = ~(df2['j_call'].str.contains('J'))
	df2 = df2.drop(df2[cond].index.values)
	
	non_n = nrow_i-df2[df2.columns[0]].count()
	#if conscount_flag:
	#   df2['consensus_count'] = df2['consensus_count'].replace(1,2)
	
	# removing sequences with low cons count
	
	filter_column = "duplicate_count"
	if conscount_flag: filter_column = "consensus_count"
	df_cons_low = df2[df2[filter_column]<conscount_filter]
	nrow_i = df2[df2.columns[0]].count()
	df2 = df2[df2[filter_column]>=conscount_filter]
	
	
	cons_n = str(nrow_i-df2[df2.columns[0]].count())
	nrow_i = df2[df2.columns[0]].count()    
	
	df2.to_csv('${outfile}'+'passed.tsv', sep = '\t',index=False) #, compression='gzip'
	
	pd.concat([df_cons_low,df_non]).to_csv('${outfile}'+'failed.tsv', sep = '\t',index=False)
	
	print(str(low_n)+' Sequences had N count over 10')
	print(str(dup_n)+' Sequences had a duplicated sequnece id')
	print(str(collapse_n)+' Sequences were collapsed')
	print(str(df_non[df_non.columns[0]].count())+' Sequences were declared non functional or lacked a J assignment')
	#print(str(df_cons_low[df_cons_low.columns[0]].count())+' Sequences had a '+filter_column+' lower than threshold')
	print('Going forward with '+str(df2[df2.columns[0]].count())+' sequences')
	
	"""
}else{
	"""
	
	"""
}

}


process airrseq_to_fasta {

input:
 set val(name), file(airrseq_data) from g0_19_outputFileTSV0_g_80

output:
 set val(name), file(outfile)  into g_80_germlineFastaFile0_g11_12, g_80_germlineFastaFile0_g11_9, g_80_germlineFastaFile0_g21_12, g_80_germlineFastaFile0_g21_9

script:

outfile = name+"_collapsed_seq.fasta"

"""
#!/usr/bin/env Rscript

data <- data.table::fread("${airrseq_data}", stringsAsFactors = F, data.table = F)

data_columns <- names(data)

# take extra columns after cdr3

idx_cdr <- which(data_columns=="cdr3")+1

add_columns <- data_columns[idx_cdr:length(data_columns)]

unique_information <- unique(c("sequence_id", "duplicate_count", "consensus_count", "c_call", add_columns))

unique_information <- unique_information[unique_information %in% data_columns]

seqs <- data[["sequence"]]

seqs_name <-
  sapply(1:nrow(data), function(x) {
    paste0(unique_information,
           rep('=', length(unique_information)),
           data[x, unique_information],
           collapse = '|')
  })
seqs_name <- gsub('sequence_id=', '', seqs_name, fixed = T)

tigger::writeFasta(setNames(seqs, seqs_name), "${outfile}")

"""
}

if(params.container.startsWith("peresay")){
	cmd = 'source("/usr/local/bin/functions_tigger.R")'
}else{
	cmd = 'library(tigger)'
}
process Undocumented_Alleles {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*novel-passed.tsv$/) "novel_report/$filename"}
input:
 set val(name),file(airr_file) from g0_19_outputFileTSV0_g_8
 set val(v_germline_name), file(v_germline_file) from g_2_germlineFastaFile_g_8

output:
 set val(name),file("*novel-passed.tsv") optional true  into g_8_outputFileTSV00
 set val("v_germline"), file("V_novel_germline.fasta") optional true  into g_8_germlineFastaFile1_g_70

script:
chain = params.Undocumented_Alleles.chain
num_threads = params.Undocumented_Alleles.num_threads
germline_min = params.Undocumented_Alleles.germline_min
min_seqs = params.Undocumented_Alleles.min_seqs
auto_mutrange = params.Undocumented_Alleles.auto_mutrange
mut_range = params.Undocumented_Alleles.mut_range
pos_range = params.Undocumented_Alleles.pos_range
y_intercept = params.Undocumented_Alleles.y_intercept
alpha = params.Undocumented_Alleles.alpha
j_max = params.Undocumented_Alleles.j_max
min_frac = params.Undocumented_Alleles.min_frac


out_novel_file = airr_file.toString() - ".tsv" + "_novel-passed.tsv"

out_novel_germline = "V_novel_germline"

"""
#!/usr/bin/env Rscript

${cmd}

# libraries
suppressMessages(require(dplyr))

# functions

## check for repeated nucliotide in sequece. get the novel allele and the germline sequence.
Repeated_Read <- function(x, seq) {
  NT <- as.numeric(gsub('([0-9]+).*', '\\1', x))
  SNP <- gsub('.*>', '', x)
  OR_SNP <- gsub('[0-9]+([[:alpha:]]*).*', '\\1', x)
  seq <- c(substr(seq, (NT), (NT + 3)),
           substr(seq, (NT - 1), (NT + 2)),
           substr(seq, (NT - 2), (NT + 1)),
           substr(seq, (NT - 3), (NT)))
  PAT <- paste0(c(
    paste0(c(rep(SNP, 3), OR_SNP), collapse = ""),
    paste0(c(rep(SNP, 2), OR_SNP, SNP), collapse = ""),
    paste0(c(SNP, OR_SNP, rep(SNP, 2)), collapse = ""),
    paste0(c(OR_SNP, rep(SNP, 3)), collapse = "")
  ), collapse = '|')
  if (any(grepl(PAT, seq)))
    return(gsub(SNP, 'X', gsub(OR_SNP, 'z', seq[grepl(PAT, seq)])))
  else
    return(NA)
}

# read data and germline
data <- data.table::fread('${airr_file}', stringsAsFactors = F, data.table = F)
vgerm <- tigger::readIgFasta('${v_germline_file}')

# transfer groovy param to rsctipt
num_threads = as.numeric(${num_threads})
germline_min = as.numeric(${germline_min})
min_seqs = as.numeric(${min_seqs})
y_intercept = as.numeric(${y_intercept})
alpha = as.numeric(${alpha})
j_max = as.numeric(${j_max})
min_frac = as.numeric(${min_frac})
auto_mutrange = as.logical('${auto_mutrange}')
mut_range = as.numeric(unlist(strsplit('${mut_range}',":")))
mut_range = mut_range[1]:mut_range[2]
pos_range = as.numeric(unlist(strsplit('${pos_range}',":")))
pos_range = pos_range[1]:pos_range[2]


novel =  try(findNovelAlleles(
data = data,
germline_db = vgerm,
v_call = 'v_call',
j_call = 'j_call' ,
seq = 'sequence_alignment',
junction = 'junction',
junction_length = 'junction_length',
germline_min = germline_min,
min_seqs = min_seqs,
y_intercept = y_intercept,
alpha = alpha,
j_max = j_max,
min_frac = min_frac,
auto_mutrange = auto_mutrange,
mut_range = mut_range,
pos_range = pos_range,
nproc = num_threads
))
	
  
# select only the novel alleles
if (class(novel) != 'try-error') {

	if (nrow(novel) != 0) {
		novel <- tigger::selectNovel(novel)
		novel <- novel %>% dplyr::distinct(novel_imgt, .keep_all = TRUE) %>% 
		dplyr::filter(!is.na(novel_imgt), nt_substitutions!='') %>% 
		dplyr::mutate(gene = alakazam::getGene(germline_call, strip_d = F)) %>%
		dplyr::group_by(gene) %>% dplyr::top_n(n = 2, wt = novel_imgt_count)
	}
	
	## remove padded alleles
	print(novel)
	
	if (nrow(novel) != 0) {
		SNP_XXXX <- unlist(sapply(1:nrow(novel), function(i) {
		  subs <- strsplit(novel[['nt_substitutions']][i], ',')[[1]]
		  RR <-
		    unlist(sapply(subs,
		           Repeated_Read,
		           seq = novel[['germline_imgt']][i],
		           simplify = F))
		  RR <- RR[!is.na(RR)]
		  
		  length(RR) != 0
		}))
		
		novel <- novel[!SNP_XXXX, ]
		
		# save novel output
		write.table(
		    novel,
		    file = '${out_novel_file}',
		    row.names = FALSE,
		    quote = FALSE,
		    sep = '\t'
		)
		
		# save germline
		novel_v_germline <- setNames(gsub('-', '.', novel[['novel_imgt']], fixed = T), novel[['polymorphism_call']])
		tigger::writeFasta(c(vgerm, novel_v_germline), paste0('${out_novel_germline}','.fasta'))
	}else{
		## write fake file
		file.copy(from = '${v_germline_file}', to = paste0('./','${out_novel_germline}','.fasta'))
		
		#file.create(paste0('${out_novel_germline}','.txt'))
		
	}
	
	
}else{
	file.copy(from = '${v_germline_file}', to = paste0('./','${out_novel_germline}','.fasta'))
	#file.create(paste0('${out_novel_germline}','.txt'))
}
"""


}

g_8_germlineFastaFile1_g_70= g_8_germlineFastaFile1_g_70.ifEmpty([""]) 


process change_names_fasta {

input:
 set val(name), file(v_ref) from g_8_germlineFastaFile1_g_70

output:
 set val(name), file("new_V_novel_germline*")  into g_70_germlineFastaFile0_g_86, g_70_germlineFastaFile0_g11_22, g_70_germlineFastaFile0_g11_12
 file "changes.csv" optional true  into g_70_outputFileCSV1_g_86


script:

readArray_v_ref = v_ref.toString().split(' ')[0]

if(readArray_v_ref.endsWith("fasta")){

"""
#!/usr/bin/env python3 

import pandas as pd
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from hashlib import sha256 


def fasta_to_dataframe(file_path):
    data = {'ID': [], 'Sequence': []}
    with open(file_path, 'r') as file:
        for record in SeqIO.parse(file, 'fasta'):
            data['ID'].append(record.id)
            data['Sequence'].append(str(record.seq))

        df = pd.DataFrame(data)
        return df


file_path = '${readArray_v_ref}'  # Replace with the actual path
df = fasta_to_dataframe(file_path)


for index, row in df.iterrows():   
  if len(row['ID']) > 50:
    print("hoo")
    print(row['ID'])
    row['ID'] = row['ID'].split('*')[0] + '*' + row['ID'].split('*')[1].split('_')[0] + '_' + sha256(row['Sequence'].encode('utf-8')).hexdigest()[-4:]


def dataframe_to_fasta(df, output_file, description_column='Description', default_description=''):
    records = []

    for index, row in df.iterrows():
        sequence_record = SeqRecord(Seq(row['Sequence']), id=row['ID'])

        # Use the description from the DataFrame if available, otherwise use the default
        description = row.get(description_column, default_description)
        sequence_record.description = description

        records.append(sequence_record)

    with open(output_file, 'w') as output_handle:
        SeqIO.write(records, output_handle, 'fasta')

def save_changes_to_csv(old_df, new_df, output_file):
    changes = []
    for index, (old_row, new_row) in enumerate(zip(old_df.itertuples(), new_df.itertuples()), 1):
        if old_row.ID != new_row.ID:
            changes.append({'Row': index, 'Old_ID': old_row.ID, 'New_ID': new_row.ID})
    
    changes_df = pd.DataFrame(changes)
    if not changes_df.empty:
        changes_df.to_csv(output_file, index=False)
        
output_file_path = 'new_V_novel_germline.fasta'

dataframe_to_fasta(df, output_file_path)


file_path = '${readArray_v_ref}'  # Replace with the actual path
old_df = fasta_to_dataframe(file_path)

output_csv_file = "changes.csv"
save_changes_to_csv(old_df, df, output_csv_file)

"""
} else{
	
"""
#!/usr/bin/env python3 
	

file_path = 'new_V_novel_germline.txt'

with open(file_path, 'w'):
    pass
    
"""    
}    
}


process Second_Alignment_V_MakeBlastDb {

input:
 set val(db_name), file(germlineFile) from g_70_germlineFastaFile0_g11_22

output:
 file "${db_name}"  into g11_22_germlineDb0_g11_9

script:

if(germlineFile.getName().endsWith("fasta")){
	"""
	sed -e '/^>/! s/[.]//g' ${germlineFile} > tmp_germline.fasta
	mkdir -m777 ${db_name}
	makeblastdb -parse_seqids -dbtype nucl -in tmp_germline.fasta -out ${db_name}/${db_name}
	"""
}else{
	"""
	echo something if off
	"""
}

}


process ogrdbstats_report_first_alignment {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*pdf$/) "ogrdbstats_first_alignment/$filename"}
publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*csv$/) "ogrdbstats_first_alignment/$filename"}
input:
 set val(name),file(airrFile) from g0_19_outputFileTSV0_g_68
 set val(name1), file(germline_file) from g0_12_germlineFastaFile1_g_68
 set val(name2), file(v_germline_file) from g_2_germlineFastaFile_g_68

output:
 file "*pdf"  into g_68_outputFilePdf00
 file "*csv"  into g_68_outputFileCSV11

script:

// general params
chain = params.ogrdbstats_report_first_alignment.chain
outname = airrFile.name.toString().substring(0, airrFile.name.toString().indexOf("_db-pass"))

"""

germline_file_path=\$(realpath ${germline_file})

novel=""

if grep -q "_[A-Z][0-9]" ${v_germline_file}; then
	grep -A 6 "_[A-Z][0-9]" ${v_germline_file} > novel_sequences.fasta
	novel=\$(realpath novel_sequences.fasta)
	diff \$germline_file_path \$novel | grep '^<' | sed 's/^< //' > personal_germline.fasta
	germline_file_path=\$(realpath personal_germline.fasta)
	novel="--inf_file \$novel"
fi

IFS='\t' read -a var < ${airrFile}

airrfile=${airrFile}

if [[ ! "\${var[*]}" =~ "v_call_genotyped" ]]; then
    awk -F'\t' '{col=\$5;gsub("call", "call_genotyped", col); print \$0 "\t" col}' ${airrFile} > ${outname}_genotyped.tsv
    airrfile=${outname}_genotyped.tsv
fi

airrFile_path=\$(realpath \$airrfile)


run_ogrdbstats \
	\$germline_file_path \
	"Homosapiens" \
	\$airrFile_path \
	${chain} \
	\$novel 

"""

}


process Second_Alignment_D_MakeBlastDb {

input:
 set val(db_name), file(germlineFile) from g_3_germlineFastaFile_g11_16

output:
 file "${db_name}"  into g11_16_germlineDb0_g11_9

script:

if(germlineFile.getName().endsWith("fasta")){
	"""
	sed -e '/^>/! s/[.]//g' ${germlineFile} > tmp_germline.fasta
	mkdir -m777 ${db_name}
	makeblastdb -parse_seqids -dbtype nucl -in tmp_germline.fasta -out ${db_name}/${db_name}
	"""
}else{
	"""
	echo something if off
	"""
}

}


process Second_Alignment_J_MakeBlastDb {

input:
 set val(db_name), file(germlineFile) from g_4_germlineFastaFile_g11_17

output:
 file "${db_name}"  into g11_17_germlineDb0_g11_9

script:

if(germlineFile.getName().endsWith("fasta")){
	"""
	sed -e '/^>/! s/[.]//g' ${germlineFile} > tmp_germline.fasta
	mkdir -m777 ${db_name}
	makeblastdb -parse_seqids -dbtype nucl -in tmp_germline.fasta -out ${db_name}/${db_name}
	"""
}else{
	"""
	echo something if off
	"""
}

}


process Second_Alignment_IgBlastn {

input:
 set val(name),file(fastaFile) from g_80_germlineFastaFile0_g11_9
 file db_v from g11_22_germlineDb0_g11_9
 file db_d from g11_16_germlineDb0_g11_9
 file db_j from g11_17_germlineDb0_g11_9

output:
 set val(name), file("${outfile}") optional true  into g11_9_igblastOut0_g11_12

script:
num_threads = params.Second_Alignment_IgBlastn.num_threads
ig_seqtype = params.Second_Alignment_IgBlastn.ig_seqtype
outfmt = params.Second_Alignment_IgBlastn.outfmt
num_alignments_V = params.Second_Alignment_IgBlastn.num_alignments_V
domain_system = params.Second_Alignment_IgBlastn.domain_system
auxiliary_data = params.Second_Alignment_IgBlastn.auxiliary_data

randomString = org.apache.commons.lang.RandomStringUtils.random(9, true, true)
outname = name + "_" + randomString
outfile = (outfmt=="MakeDb") ? name+"_"+randomString+".out" : name+"_"+randomString+".tsv"
outfmt = (outfmt=="MakeDb") ? "'7 std qseq sseq btop'" : "19"

if(db_v.toString()!="" && db_d.toString()!="" && db_j.toString()!=""){
	"""
	igblastn -query ${fastaFile} \
		-germline_db_V ${db_v}/${db_v} \
		-germline_db_D ${db_d}/${db_d} \
		-germline_db_J ${db_j}/${db_j} \
		-num_alignments_V ${num_alignments_V} \
		-domain_system imgt \
		-auxiliary_data ${auxiliary_data} \
		-outfmt ${outfmt} \
		-num_threads ${num_threads} \
		-out ${outfile}
	"""
}else{
	"""
	"""
}

}


process Second_Alignment_MakeDb {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*_db-pass.tsv$/) "second_rearrangement/$filename"}
publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*_db-fail.tsv$/) "second_rearrangement/$filename"}
input:
 set val(name),file(fastaFile) from g_80_germlineFastaFile0_g11_12
 set val(name_igblast),file(igblastOut) from g11_9_igblastOut0_g11_12
 set val(name1), file(v_germline_file) from g_70_germlineFastaFile0_g11_12
 set val(name2), file(d_germline_file) from g_3_germlineFastaFile_g11_12
 set val(name3), file(j_germline_file) from g_4_germlineFastaFile_g11_12

output:
 set val(name_igblast),file("*_db-pass.tsv") optional true  into g11_12_outputFileTSV0_g_86
 set val("reference_set"), file("${reference_set}") optional true  into g11_12_germlineFastaFile11
 set val(name_igblast),file("*_db-fail.tsv") optional true  into g11_12_outputFileTSV22

script:

failed = params.Second_Alignment_MakeDb.failed
format = params.Second_Alignment_MakeDb.format
regions = params.Second_Alignment_MakeDb.regions
extended = params.Second_Alignment_MakeDb.extended
asisid = params.Second_Alignment_MakeDb.asisid
asiscalls = params.Second_Alignment_MakeDb.asiscalls
inferjunction = params.Second_Alignment_MakeDb.inferjunction
partial = params.Second_Alignment_MakeDb.partial
name_alignment = params.Second_Alignment_MakeDb.name_alignment

failed = (failed=="true") ? "--failed" : ""
format = (format=="changeo") ? "--format changeo" : ""
extended = (extended=="true") ? "--extended" : ""
regions = (regions=="rhesus-igl") ? "--regions rhesus-igl" : ""
asisid = (asisid=="true") ? "--asis-id" : ""
asiscalls = (asiscalls=="true") ? "--asis-calls" : ""
inferjunction = (inferjunction=="true") ? "--infer-junction" : ""
partial = (partial=="true") ? "--partial" : ""

reference_set = "reference_set_makedb_"+name_alignment+".fasta"

outname = name_igblast+'_'+name_alignment

if(igblastOut.getName().endsWith(".out")){
	"""
	
	cat ${v_germline_file} ${d_germline_file} ${j_germline_file} > ${reference_set}
	
	MakeDb.py igblast \
		-s ${fastaFile} \
		-i ${igblastOut} \
		-r ${v_germline_file} ${d_germline_file} ${j_germline_file} \
		--log MD_${name}.log \
		--outname ${outname}\
		${extended} \
		${failed} \
		${format} \
		${regions} \
		${asisid} \
		${asiscalls} \
		${inferjunction} \
		${partial}
	"""
}else{
	"""
	
	"""
}

}


process change_light_germline_file_and_repertoire_file_names_back {

input:
 file csv from g_70_outputFileCSV1_g_86
 set val(name1), file(germline_file) from g_70_germlineFastaFile0_g_86
 set val(name_igblast),file(rep_file) from g11_12_outputFileTSV0_g_86

output:
 set val("${germline}"),file("${germline}")  into g_86_germlineFastaFile0_g_29
 set val("${rep}"), file("${rep}")  into g_86_outputFileTSV1_g_29, g_86_outputFileTSV1_g_31, g_86_outputFileTSV1_g_75


script:


germline = germline_file.toString().split(' ')[0]
rep = rep_file.toString().split(' ')[0]
changes_csv = csv.toString().split(' ')[0]


"""

#!/usr/bin/env Rscript


# Check if changes.csv file exists
if (file.exists("changes.csv")) {

  # Read changes from CSV
  changes <- read.csv("changes.csv", header = FALSE, col.names = c("row", "old_id", "new_id"))

  # Process changes and modify TSV files
  for (change in 1:nrow(changes)) {
  
  
    old_id <- changes[change,"old_id"]
    new_id <- changes[change,"new_id"]
    
    asterisk_pos <- gregexpr("*", old_id, fixed = TRUE)[[1]]
    old_id <- substring(old_id, asterisk_pos[1] + 1)
    
    asterisk_pos <- gregexpr("*", new_id, fixed = TRUE)[[1]]
    new_id <- substring(new_id, asterisk_pos[1] + 1)

    
    # Modify genotype file
    
    system(paste("sed -i 's/", new_id, "/", old_id, "/g' ${rep}", sep = ""))
    
    system(paste("sed -i 's/", new_id, "/", old_id, "/g' ${germline}", sep = ""))

  }


} else {
  cat("No changes.csv file found.")
}

"""

}

//* params.heavy_chain =  "yes"  //* @checkbox


process TIgGER_bayesian_genotype_Inference_d_call {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /${call}_genotype_report.tsv$/) "genotype_report/$filename"}
input:
 set val(name),file(airrFile) from g_86_outputFileTSV1_g_75
 set val(name1), file(germline_file) from g_3_germlineFastaFile_g_75

output:
 set val("${call}_genotype"),file("${call}_genotype_report.tsv") optional true  into g_75_outputFileTSV0_g_76
 set val("${call}_personal_reference"), file("${call}_personal_reference.fasta") optional true  into g_75_germlineFastaFile1_g21_16, g_75_germlineFastaFile1_g21_12
 set val("fake"),file("fake*") optional true  into g_75_outputFileTSV22

script:

// general params
call = params.TIgGER_bayesian_genotype_Inference_d_call.call
seq = params.TIgGER_bayesian_genotype_Inference_d_call.seq
find_unmutated = params.TIgGER_bayesian_genotype_Inference_d_call.find_unmutated
single_assignments = params.TIgGER_bayesian_genotype_Inference_d_call.single_assignments
germline_file = germline_file.name.startsWith('NO_FILE') ? "" : "${germline_file}"



if (params.heavy_chain == "yes"){
	"""
	#!/usr/bin/env Rscript
	
	library(tigger)
	library(data.table)
	
	## get genotyped alleles
	GENOTYPED_ALLELES <- function(y) {
	  m <- which.max(as.numeric(y[2:5]))
	  paste0(unlist(strsplit((y[1]), ','))[1:m], collapse = ",")
	}
	
	# read data
	data <- fread("${airrFile}", data.table=FALSE)
	find_unmutated_ <- "${find_unmutated}"=="true"
	germline_db <- if("${germline_file}"!="") readIgFasta("${germline_file}") else NA
	
	# get the params based on the call column
	
	params <- list("v_call" = c(0.6, 0.4, 0.4, 0.35, 0.25, 0.25, 0.25, 0.25, 0.25),
				   "d_call" = c(0.5, 0.5, 0, 0, 0, 0, 0, 0, 0),
				   "j_call" = c(0.5, 0.5, 0, 0, 0, 0, 0, 0, 0))
	
	if("${single_assignments}"=="true"){
		data <- data[!grepl(pattern = ',', data[["${call}"]]),]
	}
	
	# remove rows where there are missing values in the call column
	
	data <- data[!is.na(data[["${call}"]]),]
	
	# infer the genotype using tigger
	geno <-
	      tigger::inferGenotypeBayesian(
	        data,
	        find_unmutated = find_unmutated_,
	        germline_db = germline_db,
	        v_call = "${call}",
	        seq = "${seq}",
	        priors = params[["${call}"]]
	      )
	
	print(geno)
	
	geno[["genotyped_alleles"]] <-
	  apply(geno[, c(2, 6:9)], 1, function(y) {
	    GENOTYPED_ALLELES(y)
	  })
	
	# write the report
	write.table(geno, file = paste0("${call}","_genotype_report.tsv"), row.names = F, sep = "\t")
	
	# create the personal reference set
	NOTGENO.IND <- !(sapply(strsplit(names(germline_db), '*', fixed = T), '[', 1) %in%  geno[["gene"]])
	germline_db_new <- germline_db[NOTGENO.IND]
	
	for (i in 1:nrow(geno)) {
	  gene <- geno[i, "gene"]
	  alleles <- geno[i, "genotyped_alleles"]
	  if(alleles=="") alleles <- geno[i, "alleles"]
	  alleles <- unlist(strsplit(alleles, ','))
	  IND <- names(germline_db) %in%  paste(gene, alleles, sep = '*')
	  germline_db_new <- c(germline_db_new, germline_db[IND])
	}
	
	# writing imgt gapped fasta reference
	writeFasta(germline_db_new, file = paste0("${call}","_personal_reference.fasta"))
	
	"""
}else{

	"""
	#!/usr/bin/env Rscript
	
	library(tigger)
	library(data.table)
	
	## get genotyped alleles
	GENOTYPED_ALLELES <- function(y) {
	  m <- which.max(as.numeric(y[2:5]))
	  paste0(unlist(strsplit((y[1]), ','))[1:m], collapse = ",")
	}
	
	# read data
	data <- fread("${airrFile}", data.table=FALSE)
	find_unmutated_ <- "${find_unmutated}"=="true"
	germline_db <- if("${germline_file}"!="") readIgFasta("${germline_file}") else NA
	
	# writing imgt gapped fasta reference
	writeFasta(germline_db, file = paste0("${call}","_personal_reference.fasta"))
	
	"""
}


}


process Third_Alignment_D_MakeBlastDb {

input:
 set val(db_name), file(germlineFile) from g_75_germlineFastaFile1_g21_16

output:
 file "${db_name}"  into g21_16_germlineDb0_g21_9

script:

if(germlineFile.getName().endsWith("fasta")){
	"""
	sed -e '/^>/! s/[.]//g' ${germlineFile} > tmp_germline.fasta
	mkdir -m777 ${db_name}
	makeblastdb -parse_seqids -dbtype nucl -in tmp_germline.fasta -out ${db_name}/${db_name}
	"""
}else{
	"""
	echo something if off
	"""
}

}


process TIgGER_bayesian_genotype_Inference_j_call {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /${call}_genotype_report.tsv$/) "genotype_report/$filename"}
input:
 set val(name),file(airrFile) from g_86_outputFileTSV1_g_31
 set val(name1), file(germline_file) from g_4_germlineFastaFile_g_31

output:
 set val("${call}_genotype"),file("${call}_genotype_report.tsv")  into g_31_outputFileTSV0_g_76
 set val("${call}_personal_reference"), file("${call}_personal_reference.fasta")  into g_31_germlineFastaFile1_g21_17, g_31_germlineFastaFile1_g21_12

script:

// general params
call = params.TIgGER_bayesian_genotype_Inference_j_call.call
seq = params.TIgGER_bayesian_genotype_Inference_j_call.seq
find_unmutated = params.TIgGER_bayesian_genotype_Inference_j_call.find_unmutated
single_assignments = params.TIgGER_bayesian_genotype_Inference_j_call.single_assignments

germline_file = germline_file.name.startsWith('NO_FILE') ? "" : "${germline_file}"


"""
#!/usr/bin/env Rscript

library(tigger)
library(data.table)

## get genotyped alleles
GENOTYPED_ALLELES <- function(y) {
  m <- which.max(as.numeric(y[2:5]))
  paste0(unlist(strsplit((y[1]), ','))[1:m], collapse = ",")
}

# read data
data <- fread("${airrFile}", data.table=FALSE)
find_unmutated_ <- "${find_unmutated}"=="true"
germline_db <- if("${germline_file}"!="") readIgFasta("${germline_file}") else NA

# get the params based on the call column

params <- list("v_call" = c(0.6, 0.4, 0.4, 0.35, 0.25, 0.25, 0.25, 0.25, 0.25),
			   "d_call" = c(0.5, 0.5, 0, 0, 0, 0, 0, 0, 0),
			   "j_call" = c(0.5, 0.5, 0, 0, 0, 0, 0, 0, 0))

if("${single_assignments}"=="true"){
	data <- data[!grepl(pattern = ',', data[["${call}"]]),]
}

# remove rows where there are missing values in the call column

data <- data[!is.na(data[["${call}"]]),]

# infer the genotype using tigger
geno <-
      tigger::inferGenotypeBayesian(
        data,
        find_unmutated = find_unmutated_,
        germline_db = germline_db,
        v_call = "${call}",
        seq = "${seq}",
        priors = params[["${call}"]]
      )

print(geno)

geno[["genotyped_alleles"]] <-
  apply(geno[, c(2, 6:9)], 1, function(y) {
    GENOTYPED_ALLELES(y)
  })

# write the report
write.table(geno, file = paste0("${call}","_genotype_report.tsv"), row.names = F, sep = "\t")

# create the personal reference set
NOTGENO.IND <- !(sapply(strsplit(names(germline_db), '*', fixed = T), '[', 1) %in%  geno[["gene"]])
germline_db_new <- germline_db[NOTGENO.IND]

for (i in 1:nrow(geno)) {
  gene <- geno[i, "gene"]
  alleles <- geno[i, "genotyped_alleles"]
  if(alleles=="") alleles <- geno[i, "alleles"]
  alleles <- unlist(strsplit(alleles, ','))
  IND <- names(germline_db) %in%  paste(gene, alleles, sep = '*')
  germline_db_new <- c(germline_db_new, germline_db[IND])
}

# writing imgt gapped fasta reference
writeFasta(germline_db_new, file = paste0("${call}","_personal_reference.fasta"))

"""

}


process Third_Alignment_J_MakeBlastDb {

input:
 set val(db_name), file(germlineFile) from g_31_germlineFastaFile1_g21_17

output:
 file "${db_name}"  into g21_17_germlineDb0_g21_9

script:

if(germlineFile.getName().endsWith("fasta")){
	"""
	sed -e '/^>/! s/[.]//g' ${germlineFile} > tmp_germline.fasta
	mkdir -m777 ${db_name}
	makeblastdb -parse_seqids -dbtype nucl -in tmp_germline.fasta -out ${db_name}/${db_name}
	"""
}else{
	"""
	echo something if off
	"""
}

}


process TIgGER_bayesian_genotype_Inference_v_call {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /${call}_genotype_report.tsv$/) "genotype_report/$filename"}
input:
 set val(name),file(airrFile) from g_86_outputFileTSV1_g_29
 set val(name1), file(germline_file) from g_86_germlineFastaFile0_g_29

output:
 set val("${call}_genotype"),file("${call}_genotype_report.tsv")  into g_29_outputFileTSV0_g_76
 set val("${call}_personal_reference"), file("${call}_personal_reference.fasta")  into g_29_germlineFastaFile1_g_37, g_29_germlineFastaFile1_g_84

script:

// general params
call = params.TIgGER_bayesian_genotype_Inference_v_call.call
seq = params.TIgGER_bayesian_genotype_Inference_v_call.seq
find_unmutated = params.TIgGER_bayesian_genotype_Inference_v_call.find_unmutated
single_assignments = params.TIgGER_bayesian_genotype_Inference_v_call.single_assignments

germline_file = germline_file.name.startsWith('NO_FILE') ? "" : "${germline_file}"


"""
#!/usr/bin/env Rscript

library(tigger)
library(data.table)

## get genotyped alleles
GENOTYPED_ALLELES <- function(y) {
  m <- which.max(as.numeric(y[2:5]))
  paste0(unlist(strsplit((y[1]), ','))[1:m], collapse = ",")
}

# read data
data <- fread("${airrFile}", data.table=FALSE)
find_unmutated_ <- "${find_unmutated}"=="true"
germline_db <- if("${germline_file}"!="") readIgFasta("${germline_file}") else NA

# get the params based on the call column

params <- list("v_call" = c(0.6, 0.4, 0.4, 0.35, 0.25, 0.25, 0.25, 0.25, 0.25),
			   "d_call" = c(0.5, 0.5, 0, 0, 0, 0, 0, 0, 0),
			   "j_call" = c(0.5, 0.5, 0, 0, 0, 0, 0, 0, 0))

if("${single_assignments}"=="true"){
	data <- data[!grepl(pattern = ',', data[["${call}"]]),]
}

# remove rows where there are missing values in the call column

data <- data[!is.na(data[["${call}"]]),]

# infer the genotype using tigger
geno <-
      tigger::inferGenotypeBayesian(
        data,
        find_unmutated = find_unmutated_,
        germline_db = germline_db,
        v_call = "${call}",
        seq = "${seq}",
        priors = params[["${call}"]]
      )

print(geno)

geno[["genotyped_alleles"]] <-
  apply(geno[, c(2, 6:9)], 1, function(y) {
    GENOTYPED_ALLELES(y)
  })

# write the report
write.table(geno, file = paste0("${call}","_genotype_report.tsv"), row.names = F, sep = "\t")

# create the personal reference set
NOTGENO.IND <- !(sapply(strsplit(names(germline_db), '*', fixed = T), '[', 1) %in%  geno[["gene"]])
germline_db_new <- germline_db[NOTGENO.IND]

for (i in 1:nrow(geno)) {
  gene <- geno[i, "gene"]
  alleles <- geno[i, "genotyped_alleles"]
  if(alleles=="") alleles <- geno[i, "alleles"]
  alleles <- unlist(strsplit(alleles, ','))
  IND <- names(germline_db) %in%  paste(gene, alleles, sep = '*')
  germline_db_new <- c(germline_db_new, germline_db[IND])
}

# writing imgt gapped fasta reference
writeFasta(germline_db_new, file = paste0("${call}","_personal_reference.fasta"))

"""

}

g_29_germlineFastaFile1_g_84= g_29_germlineFastaFile1_g_84.ifEmpty([""]) 


process sec_change_names_fasta {

input:
 set val(name), file(v_ref) from g_29_germlineFastaFile1_g_84

output:
 set val(name), file("new_V_novel_germline*")  into g_84_germlineFastaFile0_g21_22, g_84_germlineFastaFile0_g21_12
 file "changes.csv" optional true  into g_84_outputFileCSV1_g_85


script:

readArray_v_ref = v_ref.toString().split(' ')[0]

if(readArray_v_ref.endsWith("fasta")){

"""
#!/usr/bin/env python3 

import pandas as pd
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from hashlib import sha256 


def fasta_to_dataframe(file_path):
    data = {'ID': [], 'Sequence': []}
    with open(file_path, 'r') as file:
        for record in SeqIO.parse(file, 'fasta'):
            data['ID'].append(record.id)
            data['Sequence'].append(str(record.seq))

        df = pd.DataFrame(data)
        return df


file_path = '${readArray_v_ref}'  # Replace with the actual path
df = fasta_to_dataframe(file_path)


for index, row in df.iterrows():   
  if len(row['ID']) > 50:
    print("hoo")
    print(row['ID'])
    row['ID'] = row['ID'].split('*')[0] + '*' + row['ID'].split('*')[1].split('_')[0] + '_' + sha256(row['Sequence'].encode('utf-8')).hexdigest()[-4:]


def dataframe_to_fasta(df, output_file, description_column='Description', default_description=''):
    records = []

    for index, row in df.iterrows():
        sequence_record = SeqRecord(Seq(row['Sequence']), id=row['ID'])

        # Use the description from the DataFrame if available, otherwise use the default
        description = row.get(description_column, default_description)
        sequence_record.description = description

        records.append(sequence_record)

    with open(output_file, 'w') as output_handle:
        SeqIO.write(records, output_handle, 'fasta')

def save_changes_to_csv(old_df, new_df, output_file):
    changes = []
    for index, (old_row, new_row) in enumerate(zip(old_df.itertuples(), new_df.itertuples()), 1):
        if old_row.ID != new_row.ID:
            changes.append({'Row': index, 'Old_ID': old_row.ID, 'New_ID': new_row.ID})
    
    changes_df = pd.DataFrame(changes)
    if not changes_df.empty:
        changes_df.to_csv(output_file, index=False)
        
output_file_path = 'new_V_novel_germline.fasta'

dataframe_to_fasta(df, output_file_path)


file_path = '${readArray_v_ref}'  # Replace with the actual path
old_df = fasta_to_dataframe(file_path)

output_csv_file = "changes.csv"
save_changes_to_csv(old_df, df, output_csv_file)

"""
} else{
	
"""
#!/usr/bin/env python3 
	

file_path = 'new_V_novel_germline.txt'

with open(file_path, 'w'):
    pass
    
"""    
}    
}


process Third_Alignment_V_MakeBlastDb {

input:
 set val(db_name), file(germlineFile) from g_84_germlineFastaFile0_g21_22

output:
 file "${db_name}"  into g21_22_germlineDb0_g21_9

script:

if(germlineFile.getName().endsWith("fasta")){
	"""
	sed -e '/^>/! s/[.]//g' ${germlineFile} > tmp_germline.fasta
	mkdir -m777 ${db_name}
	makeblastdb -parse_seqids -dbtype nucl -in tmp_germline.fasta -out ${db_name}/${db_name}
	"""
}else{
	"""
	echo something if off
	"""
}

}


process Third_Alignment_IgBlastn {

input:
 set val(name),file(fastaFile) from g_80_germlineFastaFile0_g21_9
 file db_v from g21_22_germlineDb0_g21_9
 file db_d from g21_16_germlineDb0_g21_9
 file db_j from g21_17_germlineDb0_g21_9

output:
 set val(name), file("${outfile}") optional true  into g21_9_igblastOut0_g21_12

script:
num_threads = params.Third_Alignment_IgBlastn.num_threads
ig_seqtype = params.Third_Alignment_IgBlastn.ig_seqtype
outfmt = params.Third_Alignment_IgBlastn.outfmt
num_alignments_V = params.Third_Alignment_IgBlastn.num_alignments_V
domain_system = params.Third_Alignment_IgBlastn.domain_system
auxiliary_data = params.Third_Alignment_IgBlastn.auxiliary_data

randomString = org.apache.commons.lang.RandomStringUtils.random(9, true, true)
outname = name + "_" + randomString
outfile = (outfmt=="MakeDb") ? name+"_"+randomString+".out" : name+"_"+randomString+".tsv"
outfmt = (outfmt=="MakeDb") ? "'7 std qseq sseq btop'" : "19"

if(db_v.toString()!="" && db_d.toString()!="" && db_j.toString()!=""){
	"""
	igblastn -query ${fastaFile} \
		-germline_db_V ${db_v}/${db_v} \
		-germline_db_D ${db_d}/${db_d} \
		-germline_db_J ${db_j}/${db_j} \
		-num_alignments_V ${num_alignments_V} \
		-domain_system imgt \
		-auxiliary_data ${auxiliary_data} \
		-outfmt ${outfmt} \
		-num_threads ${num_threads} \
		-out ${outfile}
	"""
}else{
	"""
	"""
}

}


process Third_Alignment_MakeDb {

input:
 set val(name),file(fastaFile) from g_80_germlineFastaFile0_g21_12
 set val(name_igblast),file(igblastOut) from g21_9_igblastOut0_g21_12
 set val(name1), file(v_germline_file) from g_84_germlineFastaFile0_g21_12
 set val(name2), file(d_germline_file) from g_75_germlineFastaFile1_g21_12
 set val(name3), file(j_germline_file) from g_31_germlineFastaFile1_g21_12

output:
 set val(name_igblast),file("*_db-pass.tsv") optional true  into g21_12_outputFileTSV0_g_85
 set val("reference_set"), file("${reference_set}") optional true  into g21_12_germlineFastaFile1_g_85
 set val(name_igblast),file("*_db-fail.tsv") optional true  into g21_12_outputFileTSV22

script:

failed = params.Third_Alignment_MakeDb.failed
format = params.Third_Alignment_MakeDb.format
regions = params.Third_Alignment_MakeDb.regions
extended = params.Third_Alignment_MakeDb.extended
asisid = params.Third_Alignment_MakeDb.asisid
asiscalls = params.Third_Alignment_MakeDb.asiscalls
inferjunction = params.Third_Alignment_MakeDb.inferjunction
partial = params.Third_Alignment_MakeDb.partial
name_alignment = params.Third_Alignment_MakeDb.name_alignment

failed = (failed=="true") ? "--failed" : ""
format = (format=="changeo") ? "--format changeo" : ""
extended = (extended=="true") ? "--extended" : ""
regions = (regions=="rhesus-igl") ? "--regions rhesus-igl" : ""
asisid = (asisid=="true") ? "--asis-id" : ""
asiscalls = (asiscalls=="true") ? "--asis-calls" : ""
inferjunction = (inferjunction=="true") ? "--infer-junction" : ""
partial = (partial=="true") ? "--partial" : ""

reference_set = "reference_set_makedb_"+name_alignment+".fasta"

outname = name_igblast+'_'+name_alignment

if(igblastOut.getName().endsWith(".out")){
	"""
	
	cat ${v_germline_file} ${d_germline_file} ${j_germline_file} > ${reference_set}
	
	MakeDb.py igblast \
		-s ${fastaFile} \
		-i ${igblastOut} \
		-r ${v_germline_file} ${d_germline_file} ${j_germline_file} \
		--log MD_${name}.log \
		--outname ${outname}\
		${extended} \
		${failed} \
		${format} \
		${regions} \
		${asisid} \
		${asiscalls} \
		${inferjunction} \
		${partial}
	"""
}else{
	"""
	
	"""
}

}


process change_names_back {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /${rep}$/) "rearrangements/$filename"}
input:
 file csv from g_84_outputFileCSV1_g_85
 set val(name_igblast),file(rep_file) from g21_12_outputFileTSV0_g_85
 set val(name2), file(rep_germline_file) from g21_12_germlineFastaFile1_g_85

output:
 set val("${rep}"), file("${rep}")  into g_85_outputFileTSV0_g_76, g_85_outputFileTSV0_g_37
 set val("${rep_germline}"),file("${rep_germline}")  into g_85_germlineFastaFile1_g_37


script:


rep = rep_file.toString().split(' ')[0]
changes_csv = csv.toString().split(' ')[0]
rep_germline = rep_germline_file.toString().split(' ')[0]

"""

#!/usr/bin/env Rscript


# Check if changes.csv file exists
if (file.exists("changes.csv")) {

  # Read changes from CSV
  changes <- read.csv("changes.csv", header = FALSE, col.names = c("row", "old_id", "new_id"))

  # Process changes and modify TSV files
  for (change in 1:nrow(changes)) {
  
  
    old_id <- changes[change,"old_id"]
    new_id <- changes[change,"new_id"]
    
    asterisk_pos <- gregexpr("*", old_id, fixed = TRUE)[[1]]
    old_id <- substring(old_id, asterisk_pos[1] + 1)
    
    asterisk_pos <- gregexpr("*", new_id, fixed = TRUE)[[1]]
    new_id <- substring(new_id, asterisk_pos[1] + 1)

    
    # Modify genotype file
    
    system(paste("sed -i 's/", new_id, "/", old_id, "/g' ${rep}", sep = ""))
    
    system(paste("sed -i 's/", new_id, "/", old_id, "/g' ${rep_germline}", sep = ""))
    
    
  }


} else {
  cat("No changes.csv file found.")
}

"""

}

g_75_outputFileTSV0_g_76= g_75_outputFileTSV0_g_76.ifEmpty([""]) 

def defaultIfInexistent(varName){
    try{
    	println binding.hasVariable(varName)
        varName.toString()
        println varName()
        return varName
    }catch(ex){
        return "check"//file("$baseDir/.emptyfiles/NO_FILE_1", hidden:true)
    }
}

def bindingVar(varName) {
    def optVar = binding.hasVariable(varName)//binding.variables.get(varName)
    println optVar
    if(optVar) {
    	println "pass"
        println optVar
        //will only run for global var
    }
    println "fail"
    optVar
}
process VDJbase_genotype_report {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /${outname}_genotype.tsv$/) "genotype_report/$filename"}
input:
 set val(name2),file(personal_run) from g_85_outputFileTSV0_g_76
 set val(name3),file(v_genotype) from g_29_outputFileTSV0_g_76
 set val(name4),file(d_genotype) from g_75_outputFileTSV0_g_76
 set val(name5),file(j_genotype) from g_31_outputFileTSV0_g_76

output:
 set val(outname),file("${outname}_genotype.tsv") optional true  into g_76_outputFileTSV00

script:

outname = initial_run.name.substring(0, initial_run.name.indexOf("_db-pass"))

"""
#!/usr/bin/env Rscript

library(dplyr)
library(data.table)
library(alakazam)

# the function get the alleles calls frequencies
getFreq <- function(data, call = "v_call"){
	# get the single assignment frequency of the alleles
	table(grep(",", data[[call]][data[[call]]!=""], invert = T, value = T))
}

addFreqInfo <- function(tab, gene, alleles){
	paste0(tab[paste0(gene, "*", unlist(strsplit(alleles, ',')))], collapse = ";")
}

## read selected data columns

data_initial_run <- fread("${initial_run}", data.table = FALSE, select = c("sequence_id", "v_call", "d_call", "j_call"))
data_genotyped <- fread("${personal_run}", data.table = FALSE, select = c("sequence_id", "v_call", "d_call", "j_call"))

## make sure that both datasets have the same sequences. 
data_initial_run <- data_initial_run[data_initial_run[["sequence_id"]] %in% data_genotyped[["sequence_id"]],]
data_genotyped <- data_genotyped[data_genotyped[["sequence_id"]] %in% data_initial_run[["sequence_id"]],]
data_initial_run <- data_initial_run[order(data_initial_run[["sequence_id"]]), ]
data_genotyped <- data_genotyped[order(data_genotyped[["sequence_id"]]), ]

non_match_v <- which(data_initial_run[["v_call"]]!=data_genotyped[["v_call"]])

data_initial_run[["v_call"]][non_match_v] <- data_genotyped[["v_call"]][non_match_v]
    

# for the v_calls
print("v_call_fractions")
tab_freq_v <- getFreq(data_genotyped, call = "v_call")
tab_clone_v <- getFreq(data_initial_run, call = "v_call")
# keep just alleles that passed the genotype
tab_clone_v <- tab_clone_v[names(tab_freq_v)]
# read the genotype table
genoV <- fread("${v_genotype}", data.table = FALSE, colClasses = "character")
# add information to the genotype table
genoV <-
  genoV %>% dplyr::group_by(gene) %>% dplyr::mutate(
    Freq_by_Clone = addFreqInfo(tab_clone_v, gene, genotyped_alleles),
    Freq_by_Seq = addFreqInfo(tab_freq_v, gene, genotyped_alleles)
  )


# for the j_calls
print("j_call_fractions")
tab_freq_j <- getFreq(data_genotyped, call = "j_call")
tab_clone_j <- getFreq(data_initial_run, call = "j_call")
# keep just alleles that passed the genotype
tab_clone_j <- tab_clone_j[names(tab_freq_j)]
# read the genotype table
genoJ <- fread("${j_genotype}", data.table = FALSE, colClasses = "character")
# add information to the genotype table
genoJ <-
  genoJ %>% dplyr::group_by(gene) %>% dplyr::mutate(
    Freq_by_Clone = addFreqInfo(tab_clone_j, gene, genotyped_alleles),
    Freq_by_Seq = addFreqInfo(tab_freq_j, gene, genotyped_alleles)
  )
  
# for the d_calls; first check if the genotype file for d exists
# if("${d_genotype}"=="*tsv")
if (endsWith("${d_genotype}", ".tsv")){
	# for the d_calls
	print("d_call_fractions")
	tab_freq_d <- getFreq(data_genotyped, call = "d_call")
	tab_clone_d <- getFreq(data_initial_run, call = "d_call")
	# keep just alleles that passed the genotype
	tab_clone_d <- tab_clone_d[names(tab_freq_d)]
	# read the genotype table
	genoD <- fread("${d_genotype}", data.table = FALSE, colClasses = "character")
	# add information to the genotype table
	print(tab_clone_d)
	print(tab_freq_d)
	print(genoD)
	genoD <-
	  genoD %>% dplyr::group_by(gene) %>% dplyr::mutate(
	    Freq_by_Clone = addFreqInfo(tab_clone_d, gene, genotyped_alleles),
	    Freq_by_Seq = addFreqInfo(tab_freq_d, gene, genotyped_alleles)
	  )
	  
	genos <- plyr::rbind.fill(genoV, genoD, genoJ)
}else{
	genos <- plyr::rbind.fill(genoV, genoJ)
}

genos[["Freq_by_Clone"]] <- gsub("NA", "0", genos[["Freq_by_Clone"]])
genos[["Freq_by_Seq"]] <- gsub("NA", "0", genos[["Freq_by_Seq"]])

# rename the genotyped_allele columns
new_genotyped_allele_name = "GENOTYPED_ALLELES"
col_loc = which(names(genos)=='genotyped_alleles')
names(genos)[col_loc] = new_genotyped_allele_name


# write the report
write.table(genos, file = paste0("${outname}","_genotype.tsv"), row.names = F, sep = "\t")
"""
}


process ogrdbstats_report {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*pdf$/) "ogrdbstats_third_alignment/$filename"}
publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*csv$/) "ogrdbstats_third_alignment/$filename"}
input:
 set val(name),file(airrFile) from g_85_outputFileTSV0_g_37
 set val(name1), file(germline_file) from g_85_germlineFastaFile1_g_37
 set val(name2), file(v_germline_file) from g_29_germlineFastaFile1_g_37

output:
 file "*pdf"  into g_37_outputFilePdf00
 file "*csv"  into g_37_outputFileCSV11

script:

// general params
chain = params.ogrdbstats_report.chain
outname = airrFile.name.toString().substring(0, airrFile.name.toString().indexOf("_db-pass"))

"""

germline_file_path=\$(realpath ${germline_file})

novel=""

if grep -q "_[A-Z][0-9]" ${v_germline_file}; then
	grep -A 6 "_[A-Z][0-9]" ${v_germline_file} > novel_sequences.fasta
	novel=\$(realpath novel_sequences.fasta)
	diff \$germline_file_path \$novel | grep '^<' | sed 's/^< //' > personal_germline.fasta
	germline_file_path=\$(realpath personal_germline.fasta)
	novel="--inf_file \$novel"
fi

IFS='\t' read -a var < ${airrFile}

airrfile=${airrFile}

if [[ ! "\${var[*]}" =~ "v_call_genotyped" ]]; then
    awk -F'\t' '{col=\$5;gsub("call", "call_genotyped", col); print \$0 "\t" col}' ${airrFile} > ${outname}_genotyped.tsv
    airrfile=${outname}_genotyped.tsv
fi

airrFile_path=\$(realpath \$airrfile)


run_ogrdbstats \
	\$germline_file_path \
	"Homosapiens" \
	\$airrFile_path \
	${chain} \
	\$novel 

"""

}


process new_meta_fata {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*json$/) "meta_data/$filename"}
input:
 set val(name), file(files) from g_44_fastaFile_g_73

output:
 file "*json"  into g_73_outputFile00


"""
#!/usr/bin/env Rscript

if (!requireNamespace("jsonlite", quietly = TRUE)) {
  install.packages("jsonlite")
}
library(jsonlite)


json_data <- list(
  sample = list(
    data_processing = list(
      annotation = list(
        aligner = list(
          tool = "IgBLAST",
          version = "1.20.0"
        ),
        aligner_reference = list(
          aligner_reference_v = "GLDB_macaque_asc_ref - version  2023-10-29",
          aligner_reference_d = "GLDB_macaque_asc_ref - version  2023-10-29",
          aligner_reference_j = "GLDB_macaque_asc_ref - version  2023-10-29"
        ),
        Genotyper = list(
          Tool = "TIgGER",
          Version = "1.0.0"
        ),
        Haplotyper = list(
          Tool = "RAbHIT",
          Version = "0.2.0"
        ),
        `Single Assignment` = "true"
      )
    )
  )
)

# Convert to JSON string without enclosing scalar values in arrays
json_string <- toJSON(json_data, pretty = TRUE, auto_unbox = TRUE)

# Write the JSON string to a file
writeLines(json_string, "annotation_metadata.json")

"""
}


workflow.onComplete {
println "##Pipeline execution summary##"
println "---------------------------"
println "##Completed at: $workflow.complete"
println "##Duration: ${workflow.duration}"
println "##Success: ${workflow.success ? 'OK' : 'failed' }"
println "##Exit status: ${workflow.exitStatus}"
}

#!/usr/bin/perl -w

##
## (C)2019-2020 Mindaugas Margelevicius
## Institute of Biotechnology, Vilnius University
##

use strict;
use FindBin;
use lib "$FindBin::Bin";
use File::Spec;
use File::Basename;
use Getopt::Long;

my  $MYPROGNAME = basename($0);
my  $TARGETDIR = '/home/mindaugas/projects/ROPIUS0/QA-tests/FM';
$TARGETDIR = '/home/mindaugas/projects/ROPIUS0/QA-tests/TBM';
$TARGETDIR = '/home/mindaugas/projects/ROPIUS0/QA-tests/FM_ENO1';
##maximum number of predicted structures to evaluate:
my  $NUM_EVAL = 50;

my  $usage = <<EOIN;

Combine protein structure QA predictions and Rosetta per-residue energy 
scores to evaluate the final obtained prediction.
Prediction summary (evaluation) files can be given by arguments and/or 
their information can be read from STDIN.
(C)2019-2020 Mindaugas Margelevicius, Vilnius University

Usage:
$MYPROGNAME <Parameters> <predictionsum1> <predictionsum2> ...

Parameters:

-o <filename>      Output evaluation file.

-h                 This text.

EOIN

##programs:
my  $QAcombine = "$FindBin::Bin/QAcombine_plus_RosettaE.pl";

my  $OUTPUT;
my  $Fail = 0;

my  $result = GetOptions(
               'o=s'      => \$OUTPUT,
               'help|h'   => sub { print $usage; exit( 0 ); }
);

do { print $usage; $Fail = 1; }  unless $result;
do { print STDERR "ERROR: Output file exists. Leaving to avoid overwriting.\n"; $Fail = 1; } if($OUTPUT && -f $OUTPUT);

do { print( STDERR "ERROR: Program not found: $QAcombine\n"); exit(1); } unless($QAcombine && -f $QAcombine);

my ($ndx,$skip,$maxnflds,$ser)=(0,0,0,0);
my ($header,$footer,$body)=('','','');

my (%SUMMS, $pred, $F);

my $ninputs = 0;

## read STDIN first
exit(1) unless GetModelEvals('', \$maxnflds, \%SUMMS);

## read files given by arguments
for($ndx = 0; $ndx <= $#ARGV && !$Fail; $ndx++) {
    my $input = $ARGV[$ndx];
    if($input =~ /^\-+o$/) {
        $skip = 1;
        next;
    }
    do{$skip--; next} if $skip>0;
    $ninputs++;
    unless( GetModelEvals($input, \$maxnflds, \%SUMMS)) {
        $Fail = 1;
        last;
    }
}

exit(1) if($Fail);

my @models = keys %SUMMS;
my $modelreg = qr/(QA2__modelRUN(\d+)e(\d+)__full_d(\d+)_p([\d\.]+))/;
my ($resline, $finresults);

die "ERROR: No models (at least two) read." if($#models < 1);

printf("\n%d inputs read, excluding STDIN; %d total models\n",$ninputs,$#models+1);

## limit to three levels of model combination
for(my $i = 0; $i <= $#models && !$Fail; $i++ ) {
    my $modname = $models[$i];
    my @indmodels;
    ##parse the combination of models
    while($modname =~ /$modelreg/g) {
        my ($mrun, $mnum, $dst, $prb) = ($2,$3,$4,$5);
        push @indmodels, $1;
    }
    ##(fa_atr, fa_elec, hbond_sr_bb, hbond_lr_bb, hbond_bb_sc, hbond_sc, fa_dun, p_aa_pp)
    my @reterms = (3,7,9,10,11,12,21,22);##Rosetta talaris2014 energy terms of interest (being considered)
    my @cnsterms = (9,10);##NOTE: terms to be included constantly
    @reterms = (7,9,10,11,12,22);
    @cnsterms = ();
    my @resterms = grep { my $t=$_;!scalar(grep{$_==$t}@cnsterms) } @reterms;##terms excluding @cnsterms

    #for(my $w = 0.1; $w <= 0.5 && !$Fail; $w += 0.1 )
    for(my $w = 0.1; $w <= 0.9 && !$Fail; $w += 0.1 )
    {
        for(my $t1 = 0; $t1 <= $#resterms && !$Fail; $t1++) {
            my @srterms1 = sort{$a <=> $b}(@cnsterms,$resterms[$t1]);
            my $srterms1lst = join(',',@srterms1);

            unless( GetResultOfCombinedModels(
                    $QAcombine, $TARGETDIR, $NUM_EVAL, \$resline,
                    \%SUMMS, $modname, \@indmodels,  $srterms1lst, $w)) {
                $Fail = 1;
                last;
            }
            print($resline);
            $finresults .= $resline;

            for(my $t2 = $t1+1; $t2 <= $#resterms && !$Fail; $t2++) {
                my @srterms2 = sort{$a <=> $b}(@cnsterms,@resterms[($t1,$t2)]);
                my $srterms2lst = join(',',@srterms2);

                unless( GetResultOfCombinedModels(
                        $QAcombine, $TARGETDIR, $NUM_EVAL, \$resline,
                        \%SUMMS, $modname, \@indmodels,  $srterms2lst, $w)) {
                    $Fail = 1;
                    last;
                }
                print($resline);
                $finresults .= $resline;

                for(my $t3 = $t2+1; $t3 <= $#resterms && !$Fail; $t3++) {
                    my @srterms3 = sort{$a <=> $b}(@cnsterms,@resterms[($t1,$t2,$t3)]);
                    my $srterms3lst = join(',',@srterms3);

                    unless( GetResultOfCombinedModels(
                            $QAcombine, $TARGETDIR, $NUM_EVAL, \$resline,
                            \%SUMMS, $modname, \@indmodels,  $srterms3lst, $w)) {
                        $Fail = 1;
                        last;
                    }
                    print($resline);
                    $finresults .= $resline;
            
                    for(my $t4 = $t3+1; $t4 <= $#resterms && !$Fail; $t4++) {
                        my @srterms4 = sort{$a <=> $b}(@cnsterms,@resterms[($t1,$t2,$t3,$t4)]);
                        my $srterms4lst = join(',',@srterms4);

                        unless( GetResultOfCombinedModels(
                                $QAcombine, $TARGETDIR, $NUM_EVAL, \$resline,
                                \%SUMMS, $modname, \@indmodels,  $srterms4lst, $w)) {
                            $Fail = 1;
                            last;
                        }
                        print($resline);
                        $finresults .= $resline;

                        for(my $t5 = $t4+1; $t5 <= $#resterms && !$Fail; $t5++) {
                            my @srterms5 = sort{$a <=> $b}(@cnsterms,@resterms[($t1,$t2,$t3,$t4,$t5)]);
                            my $srterms5lst = join(',',@srterms5);

                            unless( GetResultOfCombinedModels(
                                    $QAcombine, $TARGETDIR, $NUM_EVAL, \$resline,
                                    \%SUMMS, $modname, \@indmodels,  $srterms5lst, $w)) {
                                $Fail = 1;
                                last;
                            }
                            print($resline);
                            $finresults .= $resline;

                            for(my $t6 = $t5+1; $t6 <= $#resterms && !$Fail; $t6++) {
                                my @srterms6 = sort{$a <=> $b}(@cnsterms,@resterms[($t1,$t2,$t3,$t4,$t5,$t6)]);
                                my $srterms6lst = join(',',@srterms6);
        
                                unless( GetResultOfCombinedModels(
                                        $QAcombine, $TARGETDIR, $NUM_EVAL, \$resline,
                                        \%SUMMS, $modname, \@indmodels,  $srterms6lst, $w)) {
                                    $Fail = 1;
                                    last;
                                }
                                print($resline);
                                $finresults .= $resline;
                            }
                        }
                    }
                }
            }
        }
    }
}


exit(1) if($Fail);

$F = \*STDOUT unless $OUTPUT;
if($OUTPUT && !open($F, ">", $OUTPUT)) {
    print(STDERR "ERROR: Failed to open file for writing: $OUTPUT\n");
    exit(1);
}
print($F $finresults) if $OUTPUT;
close($F) if $OUTPUT;

print("Done.\n") if $OUTPUT;
exit(0);

## ---------------------------------------------------------------------------

sub GetModelEvals
{
    my $filename = shift;
    my $rmaxnflds = shift;
    my $rsumms = shift;
    my $ret = 1;
    my ($fh, $inputname);
    ##
    $inputname = $filename? $filename: 'STDIN';
    $fh = \*STDIN unless $filename;
    if($filename && !open($fh, $filename)) {
        print(STDERR "ERROR: GetModelEvals: Failed to open file: $filename\n");
        return 0;
    }
    while(<$fh>) {
        my @flds = split(/\s+/);
        $$rmaxnflds = $#flds+1 unless $$rmaxnflds;
        if($$rmaxnflds != $#flds+1) {
            printf(STDERR "ERROR: GetModelEvals: Invalid #fields (=%d) for %s (%s)\n",
                $#flds+1,$flds[0],$inputname);
            $ret = 0;
            last;
        }
        if(exists $$rsumms{$flds[0]}) {
            printf(STDERR "ERROR: GetModelEvals: Model already exists for %s (%s)\n",
                $flds[0],$inputname);
            $ret = 0;
            last;
        }
        do{push @{$$rsumms{$flds[0]}{TRG}}, $flds[$_] if $flds[$_]=~/^T/} for(4..$#flds);
    }
    close($fh) if $filename;
    return $ret;
}

## ---------------------------------------------------------------------------

sub GetResultOfCombinedModels
{
    my $program = shift;##program to combine predictions
    my $targetdir = shift;##directory of targets
    my $num_eval = shift;##number of predictions to evaluate
    my $rresults = shift;##ref to results string
    my $rsumms = shift;##ref to summary evaluation results hash
    my $modname = shift;##name of model
    my $rmodelids = shift;##ref to individual model (evaluated) ids
    my $reterms = shift;##Rosetta energy terms encoded in string
    my $weight = shift;##contribution weight of Rosetta energy scores
    my $ret = 1;
    my ($trg, $mod, $combname, $rosedir, $cmdline, $output);
    my ($ln, $evalresults, @means);
    my ($c,$mean,$sd)=(0,0,0);
    my $ntargets = 0;
    if( $#$rmodelids < 0 ) {
        printf(STDERR "ERROR: GetResultOfCombinedModels: Invalid #models (=%d)\n",
            $#$rmodelids+1);
        return 0;
    }
    foreach $trg (@{$$rsumms{$modname}{TRG}}) {
        $rosedir = "${targetdir}/${trg}/TS--${trg}.rsc";##directory of Rosetta energy scores for target
        $cmdline = "${program} -d ${rosedir} -t $reterms -w $weight";
        unless(-d $rosedir) {
            printf(STDERR "ERROR: GetResultOfCombinedModels: Directory of Rosetta energy scores not found: %s\n",
                    $rosedir);
            return 0;
        }
        foreach $mod (@$rmodelids) {
            my $modfilename = "${targetdir}/${trg}/QA2--${trg}/${trg}_${mod}";
            unless(-f $modfilename) {
                printf(STDERR "ERROR: GetResultOfCombinedModels: Model results file not found: %s\n",
                    $modfilename);
                return 0;
            }
            $cmdline .= " ${modfilename}";
        }

        unless( RunCommand($cmdline, 0, \$output)) {
            printf(STDERR "ERROR: GetResultOfCombinedModels: Running combined prediction failed: %s\n",$cmdline);
            return 0;
        }

        ##{{ground truth
        my $gt_file = "${targetdir}/${trg}/${trg}.txt"; ##ground truth file
        my %gt_trgs;
        unless( open(F, $gt_file)) {
            printf(STDERR "ERROR: Failed to open $gt_file\n");
            return 0;
        }
        while(<F>) { $gt_trgs{$2} = $1 if /^\s*(\d+)\s+([^\-\s]+)/ };
        close(F);
        ##}}

        my @lines = grep(/^${trg}/, split(/\n/, $output));
        ($c,$mean,$sd)=(0,0,0);
        my @mae;
        foreach $ln (sort {(split(/\s+/,$b))[1] <=> (split(/\s+/,$a))[1]} @lines) {
            my @a = split(/\s+/,$ln);
            unless( exists $gt_trgs{$a[0]}) {
                printf(STDERR "ERROR: Target model %s not found in ground truth.\n",$a[0]);
            }
            $c++;
            push @mae, abs($gt_trgs{$a[0]}-$c);
            last if $num_eval <= $c;
        }
        unless($c) {
            printf(STDERR "ERROR: No evaluation points for target %s (%s)\n",$trg,$modname);
            return 0;
        }
        $mean += $_ foreach @mae; $mean /= $c;
        $sd += ($_-$mean)*($_-$mean) foreach @mae; $sd = sqrt($sd/$c);
        $evalresults .= sprintf("   %-5s %7.3f %7.3f",${trg},$mean,$sd);
        push @means, $mean;
        $ntargets++;
    }
    if($#means < 0) {
        printf(STDERR "ERROR: No MAE values over targets obtained (%s)\n",$modname);
        return 0;
    }

    ($c,$mean,$sd)=($ntargets,0,0);
    $mean += $_ foreach @means; $mean /= $c;
    $sd += ($_-$mean)*($_-$mean) foreach @means; $sd = sqrt($sd/$c);

    $combname = join('__',@{$rmodelids}) . "___w${weight}__${reterms}";
    $$rresults = sprintf("%-200s %7.3f %7.3f |%s\n",$combname,$mean,$sd,$evalresults);
    return $ret;
}

## ---------------------------------------------------------------------------

sub RunCommand
{
    my  $cmdline = shift;
    my  $retstatus = shift;
    my  $routput = shift;##ref

    if($cmdline) {
        $$routput = `$cmdline 2>&1` if $routput;
        system( "$cmdline" ) unless $routput;
    }

    if( $? == -1 ) {
        printf( STDERR "ERROR: Failed to execute command: $!\n" );
        return 0;
    }
    if( $? & 127 ) {
        printf( STDERR "ERROR: Command terminated with signal %d (%s coredump).\n",
            ($? & 127), ($? & 128)? 'with' : 'without' );
        return 0;
    }
    else {
        if(( $? >> 8 ) != 0 ) {
            unless( $retstatus ) {
                printf( STDERR "ERROR: Command failed and exited with status %d\n", $? >> 8 );
                return 0
            }
            return( $? >> 8 );
        }
    }
    return 1;
}


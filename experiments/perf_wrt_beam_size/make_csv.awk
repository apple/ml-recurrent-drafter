BEGIN {
    printf("beam_width, beam_length, average_num_accepted_tokens, tokens_per_sec\n");
}
/benchmark_recurrent_drafting beam_width/ {
    nc = gensub(/^.*beam_width=([0-9]+), beam_length=[0-9]+.*$/, "\\1", 1);
    cl = gensub(/^.*beam_width=[0-9]+, beam_length=([0-9]+).*$/, "\\1", 1);
    printf("%d, %d, ", nc, cl);
}

/Avg accepted tokens per step/ {
    anat = gensub(/^Avg accepted tokens per step +(.*)$/, "\\1", 1);
    printf("%f, ", anat);
}
/Tokens Per Second/ {
    tps = gensub(/^Tokens Per Second +(.*)$/, "\\1", 1);
    printf("%s\n", tps);
}

/*
 * Traffic and latency of the JACCL collectives (RDMA over Thunderbolt) of one
 * MLX rank, traced in the running process with the DTrace pid provider.
 * Run it through rdma-monitor.sh, which finds the rank and formats the output.
 *
 *   sudo dtrace -Z -s rdma_monitor.d -p PID SECONDS     (SECONDS 0 = until Ctrl-C)
 *
 * The kernel has no byte counters for the rdma_enX devices and netstat only
 * sees IP traffic, so bytes are the n_bytes passed to each collective (with 2
 * ranks, about what each rank sends to the other) and latency is the time of
 * the call, waiting for the other rank included.
 *
 * Output for rdma-monitor.sh: per second "R op calls bytes avg_us max_us" lines
 * followed by "T elapsed_ms"; at the end "E elapsed_ms", "S op calls bytes
 * avg_us min_us max_us" lines and the histograms.
 */

#pragma D option quiet
#pragma D option aggsize=4m

/*
 * One entry/return pair per operation, aggregating in the return clause under
 * a literal name: only self->t_* is allocated and freed per call. Moving the
 * aggregation to a shared clause (op code in self->, or keys from probefunc)
 * fills the DTrace dynamic variable space at ~1,500 calls/s ("dynamic variable
 * drops") and adds tens of us per call. Names are globs on the demangled C++
 * names ("::" is "??"). n_bytes: arg3 for all_sum/all_gather/sum_scatter(this,
 * in, out, n_bytes, ...), arg2 for send/recv(this, buf, n_bytes, peer).
 */

pid$target:libjaccl.dylib:jaccl??MeshGroup??all_sum(*:entry,
pid$target:libjaccl.dylib:jaccl??RingGroup??all_sum(*:entry
{ self->t_sum = timestamp; self->n_sum = arg3; }

pid$target:libjaccl.dylib:jaccl??MeshGroup??all_sum(*:return,
pid$target:libjaccl.dylib:jaccl??RingGroup??all_sum(*:return
/self->t_sum/
{
	this->us = (timestamp - self->t_sum) / 1000;
	@c["all_sum"] = count(); @b["all_sum"] = sum(self->n_sum);
	@a["all_sum"] = avg(this->us); @m["all_sum"] = max(this->us);
	@C["all_sum"] = count(); @B["all_sum"] = sum(self->n_sum);
	@A["all_sum"] = avg(this->us); @Lo["all_sum"] = min(this->us); @Hi["all_sum"] = max(this->us);
	@lat["all_sum"] = quantize(this->us); @size["all_sum"] = quantize(self->n_sum);
	self->t_sum = 0;
}

pid$target:libjaccl.dylib:jaccl??MeshGroup??all_gather(*:entry,
pid$target:libjaccl.dylib:jaccl??RingGroup??all_gather(*:entry
{ self->t_gat = timestamp; self->n_gat = arg3; }

pid$target:libjaccl.dylib:jaccl??MeshGroup??all_gather(*:return,
pid$target:libjaccl.dylib:jaccl??RingGroup??all_gather(*:return
/self->t_gat/
{
	this->us = (timestamp - self->t_gat) / 1000;
	@c["all_gather"] = count(); @b["all_gather"] = sum(self->n_gat);
	@a["all_gather"] = avg(this->us); @m["all_gather"] = max(this->us);
	@C["all_gather"] = count(); @B["all_gather"] = sum(self->n_gat);
	@A["all_gather"] = avg(this->us); @Lo["all_gather"] = min(this->us); @Hi["all_gather"] = max(this->us);
	@lat["all_gather"] = quantize(this->us); @size["all_gather"] = quantize(self->n_gat);
	self->t_gat = 0;
}

pid$target:libjaccl.dylib:jaccl??MeshGroup??sum_scatter(*:entry,
pid$target:libjaccl.dylib:jaccl??RingGroup??sum_scatter(*:entry
{ self->t_sct = timestamp; self->n_sct = arg3; }

pid$target:libjaccl.dylib:jaccl??MeshGroup??sum_scatter(*:return,
pid$target:libjaccl.dylib:jaccl??RingGroup??sum_scatter(*:return
/self->t_sct/
{
	this->us = (timestamp - self->t_sct) / 1000;
	@c["sum_scatter"] = count(); @b["sum_scatter"] = sum(self->n_sct);
	@a["sum_scatter"] = avg(this->us); @m["sum_scatter"] = max(this->us);
	@C["sum_scatter"] = count(); @B["sum_scatter"] = sum(self->n_sct);
	@A["sum_scatter"] = avg(this->us); @Lo["sum_scatter"] = min(this->us); @Hi["sum_scatter"] = max(this->us);
	@lat["sum_scatter"] = quantize(this->us); @size["sum_scatter"] = quantize(self->n_sct);
	self->t_sct = 0;
}

pid$target:libjaccl.dylib:jaccl??MeshGroup??send(*:entry,
pid$target:libjaccl.dylib:jaccl??RingGroup??send(*:entry
{ self->t_snd = timestamp; self->n_snd = arg2; }

pid$target:libjaccl.dylib:jaccl??MeshGroup??send(*:return,
pid$target:libjaccl.dylib:jaccl??RingGroup??send(*:return
/self->t_snd/
{
	this->us = (timestamp - self->t_snd) / 1000;
	@c["send"] = count(); @b["send"] = sum(self->n_snd);
	@a["send"] = avg(this->us); @m["send"] = max(this->us);
	@C["send"] = count(); @B["send"] = sum(self->n_snd);
	@A["send"] = avg(this->us); @Lo["send"] = min(this->us); @Hi["send"] = max(this->us);
	@lat["send"] = quantize(this->us); @size["send"] = quantize(self->n_snd);
	self->t_snd = 0;
}

pid$target:libjaccl.dylib:jaccl??MeshGroup??recv(*:entry,
pid$target:libjaccl.dylib:jaccl??RingGroup??recv(*:entry
{ self->t_rcv = timestamp; self->n_rcv = arg2; }

pid$target:libjaccl.dylib:jaccl??MeshGroup??recv(*:return,
pid$target:libjaccl.dylib:jaccl??RingGroup??recv(*:return
/self->t_rcv/
{
	this->us = (timestamp - self->t_rcv) / 1000;
	@c["recv"] = count(); @b["recv"] = sum(self->n_rcv);
	@a["recv"] = avg(this->us); @m["recv"] = max(this->us);
	@C["recv"] = count(); @B["recv"] = sum(self->n_rcv);
	@A["recv"] = avg(this->us); @Lo["recv"] = min(this->us); @Hi["recv"] = max(this->us);
	@lat["recv"] = quantize(this->us); @size["recv"] = quantize(self->n_rcv);
	self->t_rcv = 0;
}

dtrace:::BEGIN
{
	start = timestamp;
}

profile:::tick-1s
{
	printa("R %s %@d %@d %@d %@d\n", @c, @b, @a, @m);
	printf("T %d\n", (timestamp - start) / 1000000);
	trunc(@c); trunc(@b); trunc(@a); trunc(@m);
}

profile:::tick-1s
/$1 > 0 && ++secs >= $1/
{
	exit(0);
}

dtrace:::END
{
	printf("E %d\n", (timestamp - start) / 1000000);
	printa("S %s %@d %@d %@d %@d %@d\n", @C, @B, @A, @Lo, @Hi);
	printf("\nLatency per call (us):\n");
	printa(@lat);
	printf("Bytes per call:\n");
	printa(@size);
}

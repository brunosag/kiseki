<script lang="ts">
	import katex from 'katex';

	let { math, displayMode = false }: { math: string; displayMode?: boolean } = $props();

	let htmlString = $derived(
		katex.renderToString(math, {
			displayMode: displayMode,
			throwOnError: false
		})
	);
</script>

<span>
	{@html htmlString}
</span>

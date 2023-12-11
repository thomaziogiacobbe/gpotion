defmodule BFS_seq do
  def function1(node_list, graph_edges, graph_mask, updating_graph_mask, graph_visited, cost, no_of_nodes) do
    mask_true = Stream.with_index(graph_mask)
    |> Stream.filter(fn {x, _} -> x == true end)

    n = Stream.with_index(node_list)
    |> Enum.filter(fn {{_, _}, index} -> Enum.member?(mask_true, {true, index}) end)
    |> Enum.map(fn {{s, e}, i} -> {s,e} end)

    {cost, updating_graph_mask} = for_each_node(node_list, graph_edges, graph_mask, updating_graph_mask, graph_visited, cost, no_of_nodes)
  end

  defp function2(graph_mask, updating_graph_mask, graph_visited, over, no_of_nodes) do
    update_true = Stream.with_index(updating_graph_mask)
    |> Stream.filter(fn {x, _} -> x == true end)
    |> Enum.map(fn {_, i} -> i end)

    verify_graph(update_true, graph_mask, updating_graph_mask, graph_visited, over)
  end

  defp verify_graph([index | others], graph_mask, updating_graph_mask, graph_visited, over) do
    graph_mask = List.replace_at(graph_mask, index, true)
    graph_visited = List.replace_at(graph_visited, index, true)
    over = true
    updating_graph_mask = List.replace_at(updating_graph_mask, index, true)
    verify_graph(others, graph_mask, updating_graph_mask, graph_visited, over)
  end

  defp verify_graph([], graph_mask, updating_graph_mask, graph_visited, over) do
    {graph_mask, updating_graph_mask, graph_visited, over}
  end

  defp for_each_node([node | others], graph_edges, graph_mask, updating_graph_mask, graph_visited, cost, no_of_nodes) do
    {cost, updating_graph_mask} = calculate_node(node, graph_edges, graph_mask, updating_graph_mask, graph_visited, cost, no_of_nodes, elem(node, 0))
    for_each_node(others, graph_edges, graph_mask, updating_graph_mask, graph_visited, cost, no_of_nodes)
  end

  defp for_each_node([], graph_edges, graph_mask, updating_graph_mask, graph_visited, cost, no_of_nodes) do
    {cost, updating_graph_mask}
  end

  defp calculate_node(node, graph_edges, graph_mask, updating_graph_mask, graph_visited, cost, no_of_nodes, iter) when iter < (elem(node, 0) + elem(node, 1)) do
    id = Enum.at(graph_edges, iter)
    {cost, updating_graph_mask} = if !Enum.at(graph_visited, id) do
      cost = List.replace_at(cost, id, Enum.at(cost, id) + 1)
      updating_graph_mask = List.replace_at(updating_graph_mask, id, true)
      {cost, updating_graph_mask}
    else
      {cost, updating_graph_mask}
    end
    calculate_node(node, graph_edges, graph_mask, updating_graph_mask, graph_visited, cost, no_of_nodes, iter + 1)
  end

  defp calculate_node(node, graph_edges, graph_mask, updating_graph_mask, graph_visited, cost, no_of_nodes, iter) when iter == (elem(node, 0) + elem(node, 1)) do
    {cost, updating_graph_mask}
  end

  def iterate(node_list, graph_edges, graph_mask, updating_graph_mask, graph_visited, cost, no_of_nodes) do
    iterate(node_list, graph_edges, graph_mask, updating_graph_mask, graph_visited, cost, no_of_nodes, true, 1)
  end

  defp iterate(node_list, graph_edges, graph_mask, updating_graph_mask, graph_visited, cost, no_of_nodes, over, iter) when over == true do
    {cost, updating_graph_mask} = function1(node_list, graph_edges, graph_mask, updating_graph_mask, graph_visited, cost, no_of_nodes)
    {graph_mask, updating_graph_mask, graph_visited, over} = function2(graph_mask, updating_graph_mask, graph_visited, false, no_of_nodes)
    iterate(node_list, graph_edges, graph_mask, updating_graph_mask, graph_visited, cost, no_of_nodes, over, iter + 1)
  end

  defp iterate(node_list, graph_edges, graph_mask, updating_graph_mask, graph_visited, cost, no_of_nodes, over, iter) when over == false do
    {iter, cost}
  end
end

[file_input] = System.argv()

input_read_start = System.monotonic_time()

input = case File.read(file_input) do
  {:ok, content} -> content |> String.split |> Enum.map(&String.to_integer/1)
  {:error, reason} -> exit(reason)
end

no_of_nodes = Enum.at(input, 0)

nodes = Enum.slice(input, 1, no_of_nodes * 2) |> Enum.chunk_every(2)

node_list = nodes |> Enum.map(fn [x, y] -> {x, y} end)
graph_mask = List.duplicate(false, no_of_nodes)
updating_graph_mask = List.duplicate(false, no_of_nodes)
graph_visited = List.duplicate(false, no_of_nodes)

source = Enum.at(input, no_of_nodes * 2 + 1)
source = 0

graph_mask = List.replace_at(graph_mask, source, true)
graph_visited = List.replace_at(graph_visited, source, true)

edge_list_size = Enum.at(input, no_of_nodes * 2 + 2)
edges = Enum.slice(input, no_of_nodes * 2 + 3, edge_list_size * 2) |> Enum.chunk_every(2)
graph_edges = for i <- edges do Enum.at(i, 0) end
cost = List.duplicate(-1, no_of_nodes) |> List.replace_at(0, 0)

input_read_end = System.monotonic_time()
IO.puts "Input read time: #{System.convert_time_unit(input_read_end - input_read_start, :native, :millisecond)}ms"

{i, c} = BFS_seq.iterate(node_list, graph_edges, graph_mask, updating_graph_mask, graph_visited, cost, no_of_nodes)
IO.puts "#{i}"
Stream.with_index(c)
|> Stream.each(&IO.inspect/1)
|> Stream.run

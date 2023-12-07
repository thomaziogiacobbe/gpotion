defmodule BFS_seq do
  require IEx

  def function1(node_list, graph_edges, graph_mask, updating_graph_mask, graph_visited, cost, no_of_nodes) do
    mask_true = Enum.with_index(graph_mask)
    |> Stream.filter(fn {x, _} -> x == true end)

    n = Enum.with_index(node_list )
    |> Enum.filter(fn {{_, _}, index} -> Enum.member?(mask_true, {true, index}) end)
    |> Enum.map(fn {{s, e}, i} -> {s,e} end)

    {cost, updating_graph_mask} = for_each_node(node_list, graph_edges, graph_mask, updating_graph_mask, graph_visited, cost, no_of_nodes)
  end

  def function2(graph_mask, updating_graph_mask, graph_visited, over, no_of_nodes) do
    l = for x = {true, _} <- updating_graph_mask |> Stream.with_index, do: x
    {graph_mask, graph_visited, over, updating_graph_mask}
  end

  defp for_each_node([node | others], graph_edges, graph_mask, updating_graph_mask, graph_visited, cost, no_of_nodes) do
    calculate_node(node, graph_edges, graph_mask, updating_graph_mask, graph_visited, cost, no_of_nodes, elem(node, 0))
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

  def iterate(node_list, graph_edges, graph_mask, updating_graph_mask, graph_visited, cost, no_of_nodes, over, iter) when over == false do
    {cost, updating_graph_mask} = function1(node_list, graph_edges, graph_mask, updating_graph_mask, graph_visited, cost, no_of_nodes)
    {graph_mask, graph_visited, over, updating_graph_mask} = function2(graph_mask, updating_graph_mask, graph_visited, over, no_of_nodes)
    iterate(node_list, graph_edges, graph_mask, updating_graph_mask, graph_visited, cost, no_of_nodes, over, iter + 1)
  end

  def iterate(node_list, graph_edges, graph_mask, updating_graph_mask, graph_visited, cost, no_of_nodes, over, iter) when over == true do
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
d_cost = List.duplicate(-1, no_of_nodes) |> List.replace_at(0, 0)

input_read_end = System.monotonic_time()
IO.puts "Input read time: #{System.convert_time_unit(input_read_end - input_read_start, :native, :millisecond)}ms"

Stream.iterate(0, &(&1 + 1))
  |> Enum.reduce_while(1, fn _i, acc ->
    {cost, updating_graph_mask} = BFS_seq.function1(node_list, graph_edges, graph_mask, updating_graph_mask, graph_visited, d_cost, no_of_nodes)
    {graph_mask, graph_visited, over, updating_graph_mask} = BFS_seq.function2(graph_mask, updating_graph_mask, graph_visited, false, no_of_nodes)
    stop = over
    if !stop do
      {:halt, acc}
    else
      {:cont, acc + 1}
    end
  end)

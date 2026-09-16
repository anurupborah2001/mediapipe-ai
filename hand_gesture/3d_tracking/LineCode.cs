using UnityEngine;

public class LineCode : MonoBehaviour
{

  LineRenderer lineRenderer;

  public Transform origin;

  public Transform destination;

  void Start()
  {
    lineRenderer = GetComponent<LineRenderer>(); // Get the LineRenderer component
    lineRenderer.startWidth = 0.1f; // Set the width of the line
    lineRenderer.endWidth = 0.1f; // Set the width of the line

    lineRenderer.material = new Material(Shader.Find("Sprites/Default"));
    lineRenderer.useWorldSpace = true;
  } 

  void Update()
  {
    float zOffset = 0.1f;
    if (origin == null || destination == null) return;
    lineRenderer.SetPosition(0, origin.position + new Vector3(0, 0, zOffset)); // Set the starting point of the line
    lineRenderer.SetPosition(1, destination.position + new Vector3(0, 0, zOffset)); // Set the ending point of the line
  }
    
}
